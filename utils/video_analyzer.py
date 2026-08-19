"""
utils/video_analyzer.py — samples frames from a video for deepfake analysis.

Reuses the existing image pipeline per sampled frame (face extraction + CNN
inference) rather than duplicating any detection logic — app.py's
process_video() calls extract_faces_from_image() and run_inference() on each
sampled frame, same as it does for a direct image upload.
"""

from pathlib import Path

import cv2


def sample_frames(video_path, save_dir, num_frames=12):
    """
    Samples up to num_frames evenly-spaced frames from a video and saves
    them as JPGs in save_dir. Returns a list of (frame_index, file_path)
    tuples, ordered by position in the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    step = max(1, total // num_frames)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    frame_paths = []
    idx = 0
    count = 0
    while count < num_frames and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = Path(save_dir) / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append((idx, str(frame_path)))
        idx += step
        count += 1

    cap.release()
    return frame_paths


def get_video_info(video_path):
    """Basic video metadata — duration, fps, resolution. Not forensic-grade,
    just useful context for the report."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = round(total_frames / fps, 1) if fps > 0 else None
    return {
        "duration_sec": duration,
        "fps": round(fps, 1) if fps else None,
        "resolution": f"{width}x{height}" if width and height else None,
        "total_frames": total_frames,
    }
