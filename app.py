import os
import uuid
import shutil
import torch
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

from utils.model_loader import load_model, get_transforms
from utils.face_extractor import extract_faces_from_image
from utils.ela_analysis import run_ela, save_ela_image
from utils.metadata_extractor import extract_metadata
from utils.report_generator import generate_report
from utils.heatmap import GradCAM, save_heatmap
from utils.url_analyzer import fetch_and_extract_image
from utils.video_analyzer import sample_frames, get_video_info
from utils.audio_analyzer import load_audio_model, run_audio_inference

UPLOAD_FOLDER = Path("static/uploads")
RESULTS_FOLDER = Path("static/results")
ALLOWED = {"jpg", "jpeg", "png", "mp4", "avi"}
ALLOWED_AUDIO = {"wav", "mp3", "flac", "m4a", "ogg"}

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

print("[DEtecT-it] Loading image model...")
MODEL, DEVICE = load_model()
TRANSFORMS = get_transforms()
GRADCAM = GradCAM(MODEL)
print(f"[DEtecT-it] Image model ready on {DEVICE}")

print("[DEtecT-it] Loading audio model...")
AUDIO_MODEL, AUDIO_DEVICE = load_audio_model()
print(f"[DEtecT-it] Audio model ready on {AUDIO_DEVICE}")


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def allowed_audio(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO


def run_inference(face):
    tensor = TRANSFORMS(face).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, _ = MODEL(tensor)  # Xception model returns (logits, features)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    fake_prob = float(probs[1])
    real_prob = float(probs[0])
    verdict = "DEEPFAKE" if fake_prob > real_prob else "REAL"
    confidence = round(max(fake_prob, real_prob) * 100, 2)
    return verdict, confidence


def tier_from_confidence(confidence):
    if confidence < 60:
        return "low"
    elif confidence < 80:
        return "moderate"
    return "high"


def process_image(save_path, filename, session_id):
    """
    Core analysis pipeline — runs face detection, CNN inference, ELA,
    metadata forensics, Grad-CAM, and report generation on a local image
    file. Shared by /analyze (direct upload) and /analyze-url (fetched
    from a link), so there's one source of truth for detection logic
    regardless of where the image came from.
    """
    faces = extract_faces_from_image(str(save_path))
    verdicts, confidences = [], []
    for face in faces:
        v, c = run_inference(face)
        verdicts.append(v)
        confidences.append(c)

    raw_verdict = "DEEPFAKE" if verdicts.count("DEEPFAKE") >= verdicts.count("REAL") else "REAL"
    confidence = round(float(np.mean(confidences)), 2) if confidences else 0.0

    ela_data = run_ela(str(save_path))
    ela_path = RESULTS_FOLDER / f"{session_id}_ela.jpg"
    save_ela_image(ela_data["ela_image"], str(ela_path))

    meta = extract_metadata(str(save_path))

    if not confidences:
        verdict = "INCONCLUSIVE"
        confidence_tier = "low"
    else:
        confidence_tier = tier_from_confidence(confidence)
        verdict = "INCONCLUSIVE" if confidence_tier == "low" else raw_verdict

    signal_conflict = None
    if verdict == "REAL" and confidence_tier in ("moderate", "high"):
        if ela_data.get("suspicious"):
            signal_conflict = "ELA flagged compression-artifact anomalies that don't match the CNN's REAL verdict."
        elif meta.get("risk_level") == "high":
            signal_conflict = "Metadata found a specific editing-software signature that doesn't match the CNN's REAL verdict."

    heatmap_path = None
    if faces:
        top_idx = int(np.argmax(confidences))
        top_face = faces[top_idx]
        tensor = TRANSFORMS(top_face).unsqueeze(0).to(DEVICE)
        try:
            cam = GRADCAM.generate(tensor)
            heatmap_path = RESULTS_FOLDER / f"{session_id}_heatmap.jpg"
            save_heatmap(top_face, cam, str(heatmap_path), verdict=verdict, confidence=confidence)
        except Exception as e:
            print(f"[DEtecT-it] Grad-CAM generation failed: {e}")
            heatmap_path = None

    report_path = RESULTS_FOLDER / f"{session_id}_report.html"
    generate_report(
        image_filename=filename,
        verdict=verdict,
        confidence=confidence,
        ela_results=ela_data,
        metadata_results=meta,
        heatmap_path=str(heatmap_path) if heatmap_path else None,
        ela_image_path=str(ela_path),
        output_path=str(report_path),
        confidence_tier=confidence_tier,
        signal_conflict=signal_conflict,
    )

    return {
        "session_id": session_id,
        "file": filename,
        "verdict": verdict,
        "confidence": confidence,
        "confidence_tier": confidence_tier,
        "signal_conflict": signal_conflict,
        "faces_detected": len(faces),
        "ela": ela_data,
        "metadata": meta,
        "heatmap_url": f"/results/{session_id}_heatmap.jpg" if heatmap_path else None,
        "report_url": f"/results/{session_id}_report.html",
    }


def process_video(save_path, filename, session_id, num_frames=12):
    """
    Video pipeline: samples evenly-spaced frames, runs each through the
    SAME face-extraction + CNN inference used for images, then aggregates
    into one overall verdict. ELA and Grad-CAM run once, on the single
    most-confident sampled frame (not every frame).
    """
    frames_dir = UPLOAD_FOLDER / f"frames_{session_id}"
    video_info = get_video_info(save_path)
    frame_list = sample_frames(save_path, frames_dir, num_frames=num_frames)

    if not frame_list:
        shutil.rmtree(frames_dir, ignore_errors=True)
        return {
            "session_id": session_id, "file": filename, "verdict": "INCONCLUSIVE",
            "confidence": 0.0, "confidence_tier": "low", "signal_conflict": None,
            "faces_detected": 0, "ela": {}, "metadata": {}, "heatmap_url": None,
            "report_url": None, "video_note": "Could not read any frames from this video.",
        }

    frame_results = []
    for frame_idx, frame_path in frame_list:
        faces = extract_faces_from_image(frame_path)
        for face in faces:
            v, c = run_inference(face)
            frame_results.append((frame_idx, frame_path, v, c, face))

    if not frame_results:
        shutil.rmtree(frames_dir, ignore_errors=True)
        return {
            "session_id": session_id, "file": filename, "verdict": "INCONCLUSIVE",
            "confidence": 0.0, "confidence_tier": "low", "signal_conflict": None,
            "faces_detected": 0, "ela": {}, "metadata": {}, "heatmap_url": None,
            "report_url": None,
            "video_note": f"No faces detected in {len(frame_list)} sampled frames.",
        }

    verdicts = [r[2] for r in frame_results]
    confidences = [r[3] for r in frame_results]
    frames_flagged = sum(1 for v in verdicts if v == "DEEPFAKE")

    raw_verdict = "DEEPFAKE" if verdicts.count("DEEPFAKE") >= verdicts.count("REAL") else "REAL"
    confidence = round(float(np.mean(confidences)), 2)
    confidence_tier = tier_from_confidence(confidence)
    verdict = "INCONCLUSIVE" if confidence_tier == "low" else raw_verdict

    top_idx = int(np.argmax(confidences))
    top_frame_idx, top_frame_path, _, _, top_face = frame_results[top_idx]

    ela_data = run_ela(top_frame_path)
    ela_path = RESULTS_FOLDER / f"{session_id}_ela.jpg"
    save_ela_image(ela_data["ela_image"], str(ela_path))

    heatmap_path = None
    tensor = TRANSFORMS(top_face).unsqueeze(0).to(DEVICE)
    try:
        cam = GRADCAM.generate(tensor)
        heatmap_path = RESULTS_FOLDER / f"{session_id}_heatmap.jpg"
        save_heatmap(top_face, cam, str(heatmap_path), verdict=verdict, confidence=confidence)
    except Exception as e:
        print(f"[DEtecT-it] Grad-CAM generation failed: {e}")
        heatmap_path = None

    video_note = (
        f"Video analysis: {len(frame_list)} frames sampled "
        f"({video_info.get('duration_sec', '?')}s, {video_info.get('resolution', '?')}), "
        f"{frames_flagged}/{len(frame_results)} face detections flagged as DEEPFAKE. "
        f"ELA/Grad-CAM shown for the highest-confidence frame (#{top_frame_idx})."
    )

    signal_conflict = None
    if verdict == "REAL" and confidence_tier in ("moderate", "high") and ela_data.get("suspicious"):
        signal_conflict = "ELA flagged compression-artifact anomalies on the representative frame that don't match the CNN's REAL verdict."

    report_path = RESULTS_FOLDER / f"{session_id}_report.html"
    generate_report(
        image_filename=filename,
        verdict=verdict,
        confidence=confidence,
        ela_results=ela_data,
        metadata_results={},
        heatmap_path=str(heatmap_path) if heatmap_path else None,
        ela_image_path=str(ela_path),
        output_path=str(report_path),
        confidence_tier=confidence_tier,
        signal_conflict=signal_conflict,
        video_note=video_note,
    )

    shutil.rmtree(frames_dir, ignore_errors=True)

    return {
        "session_id": session_id,
        "file": filename,
        "verdict": verdict,
        "confidence": confidence,
        "confidence_tier": confidence_tier,
        "signal_conflict": signal_conflict,
        "faces_detected": len(frame_results),
        "ela": ela_data,
        "metadata": {},
        "heatmap_url": f"/results/{session_id}_heatmap.jpg" if heatmap_path else None,
        "report_url": f"/results/{session_id}_report.html",
        "video_note": video_note,
    }


def process_audio(save_path, filename, session_id):
    """
    Audio pipeline: AASIST-L voice spoof-detection model, run directly on
    the uploaded clip (no frame sampling needed — it's a fixed 4.04s
    window with tile-padding for shorter clips). No ELA/metadata/Grad-CAM
    equivalent exists for audio in this v1 — those are image-specific
    signals — so this report is intentionally lighter than the image one.
    """
    verdict_raw, confidence = run_audio_inference(AUDIO_MODEL, AUDIO_DEVICE, save_path)
    confidence_tier = tier_from_confidence(confidence)
    verdict = "INCONCLUSIVE" if confidence_tier == "low" else verdict_raw

    audio_note = (
        f"Audio analysis: AASIST-L voice spoof-detection model "
        f"(ASVspoof2019 LA, EER 0.99%). Analyzed a ~4-second window of the clip."
    )

    report_path = RESULTS_FOLDER / f"{session_id}_report.html"
    generate_report(
        image_filename=filename,
        verdict=verdict,
        confidence=confidence,
        ela_results={},
        metadata_results={},
        heatmap_path=None,
        ela_image_path=None,
        output_path=str(report_path),
        confidence_tier=confidence_tier,
        signal_conflict=None,
        video_note=audio_note,  # reusing the same report field for the note
    )

    return {
        "session_id": session_id,
        "file": filename,
        "verdict": verdict,
        "confidence": confidence,
        "confidence_tier": confidence_tier,
        "signal_conflict": None,
        "faces_detected": None,
        "ela": {},
        "metadata": {},
        "heatmap_url": None,
        "report_url": f"/results/{session_id}_report.html",
        "video_note": audio_note,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if not allowed(file.filename):
        flash("Unsupported file type.")
        return redirect(url_for("index"))

    session_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    save_path = UPLOAD_FOLDER / f"{session_id}_{filename}"
    file.save(str(save_path))

    ext = filename.rsplit(".", 1)[1].lower()
    if ext in {"mp4", "avi"}:
        results = process_video(save_path, filename, session_id)
    else:
        results = process_image(save_path, filename, session_id)
    return render_template("result.html", results=results)


@app.route("/analyze-audio", methods=["GET", "POST"])
def analyze_audio_route():
    if request.method == "GET":
        return render_template("audio_analysis.html")

    if "file" not in request.files:
        return render_template("audio_analysis.html", error="No file uploaded.")

    file = request.files["file"]
    if not allowed_audio(file.filename):
        return render_template("audio_analysis.html", error="Unsupported audio format. Use WAV, MP3, FLAC, M4A, or OGG.")

    session_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    save_path = UPLOAD_FOLDER / f"{session_id}_{filename}"
    file.save(str(save_path))

    results = process_audio(save_path, filename, session_id)
    return render_template("result.html", results=results)


@app.route("/analyze-url", methods=["GET", "POST"])
def analyze_url_route():
    if request.method == "GET":
        return render_template("url_analysis.html")

    url = request.form.get("url", "").strip()
    if not url:
        return render_template("url_analysis.html", error="Please enter a URL.")

    session_id = str(uuid.uuid4())[:8]
    local_path, filename, error = fetch_and_extract_image(url, UPLOAD_FOLDER)

    if error:
        return render_template("url_analysis.html", error=error)

    final_path = UPLOAD_FOLDER / f"{session_id}_{filename}"
    os.rename(local_path, final_path)

    results = process_image(final_path, filename, session_id)
    return render_template("result.html", results=results)


@app.route("/results/<filename>")
def serve_result(filename):
    return app.send_static_file(f"results/{filename}")


@app.route("/download/<filename>")
def download_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
