from PIL import Image, ImageChops, ImageEnhance
import io
import numpy as np

def run_ela(image_path, quality=90, amplify=20):
    original = Image.open(image_path).convert("RGB")
    
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    
    diff = ImageChops.difference(original, recompressed)
    
    diff_arr = np.array(diff).astype(float)
    mean_error = float(diff_arr.mean())
    
    ela_image = ImageEnhance.Brightness(diff).enhance(amplify)
    
    anomaly_score = min(mean_error / 15.0, 1.0)
    suspicious = anomaly_score > 0.35
    
    return {
        "ela_image"     : ela_image,
        "mean_error"    : round(mean_error, 4),
        "anomaly_score" : round(anomaly_score, 4),
        "suspicious"    : suspicious,
    }