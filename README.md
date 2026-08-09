# DEtecT-it

**Deepfake Detection & Forensic Analysis Tool**

A Flask-based web application for detecting face-reenactment deepfakes in images and generating a downloadable forensic report of the analysis. Built as an academic/portfolio project targeting Digital Forensics and Incident Response (DFIR) use cases.

## What it does

Upload an image → the tool runs three independent forensic checks and combines them into a single verdict with an explicit confidence tier, rather than a black-box "real/fake" label:

- **Deepfake classification** — Xception CNN (pretrained checkpoint from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench), trained on all four FaceForensics++ manipulation methods: DeepFakes, Face2Face, FaceSwap, NeuralTextures)
- **Error Level Analysis (ELA)** — flags compression-artifact anomalies consistent with image editing
- **EXIF metadata forensics** — flags missing/altered metadata and known editing-software signatures
- **Grad-CAM heatmap** — visualizes which facial regions the model weighted most heavily for its verdict
- **3-tier confidence system** (high / moderate / low) — a barely-above-50% prediction is surfaced as `INCONCLUSIVE` rather than a falsely confident verdict
- **Signal-conflict detection** — flags cases where ELA or metadata disagree with a confident CNN "REAL" verdict, recommending manual review
- **Forensic report export** — a standalone, downloadable HTML report per case (verdict, confidence tier, ELA/metadata findings, heatmap, methodology)

## Scope & limitations

This model detects **face-reenactment / face-swap style deepfakes** within the FaceForensics++ manipulation family. It is **not** trained to detect fully AI-generated synthetic images (e.g. Midjourney, DALL·E, Stable Diffusion outputs), and may not reliably catch deepfakes made with tools outside that family. This is stated explicitly in the app's UI and in every generated report — a "REAL" verdict means no reenactment-style manipulation was found in this scope, not a general authenticity guarantee.

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Flask (Python) |
| Model | PyTorch, Xception (DeepfakeBench pretrained weights) |
| Image forensics | OpenCV, Pillow, piexif, NumPy, SciPy |
| Visualization | Matplotlib (Grad-CAM heatmaps) |
| Frontend | HTML, CSS, vanilla JavaScript |

## Project structure

```
DEtecT-it/
├── app.py                  # Flask routes: upload, analyze, view/download report
├── utils/
│   ├── model_loader.py     # Loads Xception + transforms
│   ├── face_extractor.py   # Face detection/extraction
│   ├── ela_analysis.py     # Error Level Analysis
│   ├── metadata_extractor.py # EXIF metadata forensics
│   ├── heatmap.py          # Grad-CAM generation
│   └── report_generator.py # Standalone HTML forensic report
├── templates/               # Flask/Jinja2 HTML templates
├── static/                  # CSS, JS, uploads, generated results
├── tests/
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/anshikabisht111/DEtecT-it.git
cd DEtecT-it
pip install -r requirements.txt
python app.py
```
App runs at `http://localhost:5000`.

## Roadmap

- [ ] URL-based analysis module
- [ ] Video deepfake detection
- [ ] Audio deepfake detection
- [ ] Dedicated AI-generated/synthetic image (GAN/diffusion) detection

## Author

**Anshika Bisht** — MCA Cyber Security, Lovely Professional University
[GitHub](https://github.com/anshikabisht111)

---
*Built for academic/portfolio purposes. Not intended as a substitute for professional forensic analysis or legal evidence.*
