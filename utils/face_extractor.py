import cv2
import numpy as np
from PIL import Image

def extract_faces_from_image(image_path, max_faces=5):
    img = cv2.imread(str(image_path))
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detected = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    h, w = img.shape[:2]
    
    if len(detected) == 0:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return [pil_img]
    
    faces = []
    for (x, y, fw, fh) in detected[:max_faces]:
        pad = int(fw * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)
        
        crop = img[y1:y2, x1:x2]
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        faces.append(pil_crop)
    
    return faces