from PIL import Image
from PIL.ExifTags import TAGS


def extract_metadata(image_path):
    result = {"has_exif": False, "anomalies": [], "risk_level": "low", "raw_exif": {}}

    img = Image.open(image_path)
    exif = img._getexif()
    if not exif:
        result["anomalies"].append("No EXIF metadata found — common in AI-generated images.")
        result["risk_level"] = "medium"
        return result

    result["has_exif"] = True

    decoded = {}
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, str(tag_id))
        decoded[tag] = str(value)[:100]
    result["raw_exif"] = decoded

    software = decoded.get("Software", "").lower()
    if "photoshop" in software or "gimp" in software:
        result["anomalies"].append(f"Suspicious software tag: {decoded['Software']}")
        result["risk_level"] = "high"

    return result   