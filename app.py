import os
import uuid
import torch
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

from utils.model_loader import load_model, get_transforms
from utils.face_extractor import extract_faces_from_image
from utils.ela_analysis import run_ela, save_ela_image
from utils.metadata_extractor import extract_metadata
from utils.report_generator import generate_report
from utils.heatmap import GradCAM, save_heatmap

UPLOAD_FOLDER = Path("static/uploads")
RESULTS_FOLDER = Path("static/results")
ALLOWED = {"jpg", "jpeg", "png", "mp4", "avi"}

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

print("[DEtecT-it] Loading model...")
MODEL, DEVICE = load_model()
TRANSFORMS = get_transforms()
GRADCAM = GradCAM(MODEL)
print(f"[DEtecT-it] Ready on {DEVICE}")


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


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

    faces = extract_faces_from_image(str(save_path))
    verdicts, confidences = [], []
    for face in faces:
        v, c = run_inference(face)
        verdicts.append(v)
        confidences.append(c)

    raw_verdict = "DEEPFAKE" if verdicts.count("DEEPFAKE") >= verdicts.count("REAL") else "REAL"
    confidence = round(float(np.mean(confidences)), 2)

    # Compute ELA + metadata BEFORE the verdict tier, so we can cross-check
    # the CNN's verdict against other independent signals.
    ela_data = run_ela(str(save_path))
    ela_path = RESULTS_FOLDER / f"{session_id}_ela.jpg"
    save_ela_image(ela_data["ela_image"], str(ela_path))

    meta = extract_metadata(str(save_path))

    # Confidence-tier logic: don't show a confident REAL/DEEPFAKE badge when
    # the model itself is barely more sure than a coin flip. confidence is
    # always >= 50 (it's max(fake_prob, real_prob)*100), so <60 means the
    # model's top class barely edged out the other — that's not a reliable
    # verdict and shouldn't be presented as one.
    if confidence < 60:
        verdict = "INCONCLUSIVE"
        confidence_tier = "low"
    elif confidence < 80:
        verdict = raw_verdict
        confidence_tier = "moderate"
    else:
        verdict = raw_verdict
        confidence_tier = "high"

    # Signal-conflict check: only escalate on SPECIFIC anomalies (ELA flagged
    # suspicious, or metadata found an actual editing-software signature),
    # not the generic "no EXIF found" medium-risk flag — that fires on
    # nearly every screenshot/re-shared image regardless of authenticity,
    # so using it here would make the tool cry wolf on completely normal
    # screenshots. Only metadata's "high" tier (specific software signature)
    # counts as a real corroborating anomaly.
    signal_conflict = None
    if verdict == "REAL" and confidence_tier in ("moderate", "high"):
        if ela_data.get("suspicious"):
            signal_conflict = "ELA flagged compression-artifact anomalies that don't match the CNN's REAL verdict."
        elif meta.get("risk_level") == "high":
            signal_conflict = "Metadata found a specific editing-software signature that doesn't match the CNN's REAL verdict."

    # Grad-CAM heatmap on the most-confident detected face
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
    )

    results = {
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
    return render_template("result.html", results=results)


@app.route("/results/<filename>")
def serve_result(filename):
    return app.send_static_file(f"results/{filename}")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)