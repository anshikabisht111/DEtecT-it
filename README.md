# DEtecT-it

**Multi-Modal Deepfake Detection & Forensic Analysis Tool**

A Flask-based web application for detecting face-reenactment deepfakes and voice-spoofed audio, with URL-based analysis and downloadable forensic reports. Built as an academic/portfolio project targeting Digital Forensics and Incident Response (DFIR) use cases.

## What it does

Four analysis modes, one forensic pipeline:

- 🖼️ **Image upload** — runs three independent forensic checks and combines them into a single verdict with an explicit confidence tier, rather than a black-box "real/fake" label
- 🎞️ **Video upload** — samples frames across the clip, runs each through the same CNN pipeline as images, and aggregates into one verdict (with ELA/Grad-CAM shown for the most-confident frame)
- 🔗 **URL analysis** — paste a link, the tool fetches the image and runs it through the identical pipeline used for direct uploads
- 🎙️ **Audio upload** — voice spoof/clone detection via AASIST-L (graph attention network, ASVspoof2019-trained)

Detection signals combined per verdict:

- **Deepfake classification** — Xception CNN (pretrained checkpoint from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench), trained on all four FaceForensics++ manipulation methods: DeepFakes, Face2Face, FaceSwap, NeuralTextures)
- **Voice spoof detection** — AASIST-L (pretrained checkpoint from [clovaai/aasist](https://github.com/clovaai/aasist), ASVspoof2019 LA, EER 0.99%)
- **Error Level Analysis (ELA)** — flags compression-artifact anomalies consistent with image editing
- **EXIF metadata forensics** — flags missing/altered metadata and known editing-software signatures
- **Grad-CAM heatmap** — visualizes which facial regions the model weighted most heavily for its verdict
- **3-tier confidence system** (high / moderate / low) — a barely-above-50% prediction is surfaced as `INCONCLUSIVE` rather than a falsely confident verdict
- **Signal-conflict detection** — flags cases where ELA or metadata disagree with a confident CNN "REAL" verdict, recommending manual review
- **Forensic report export** — a standalone, downloadable HTML report per case (verdict, confidence tier, findings, methodology)

## Scope & limitations

The image/video model detects **face-reenactment / face-swap style deepfakes** within the FaceForensics++ manipulation family. It is **not** trained to detect fully AI-generated synthetic images (e.g. Midjourney, DALL·E, Stable Diffusion outputs), and may not reliably catch deepfakes made with tools outside that family. The audio model is trained on ASVspoof2019's text-to-speech/voice-conversion attacks and may not generalize equally to every modern voice-cloning tool. This is stated explicitly in the app's UI and in every generated report — a "REAL" verdict means no manipulation was found within this scope, not a general authenticity guarantee.

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Flask (Python) |
| Image/video model | PyTorch, Xception (DeepfakeBench pretrained weights) |
| Audio model | PyTorch, AASIST-L (graph attention network) |
| Image forensics | OpenCV, Pillow, piexif, NumPy, SciPy |
| Audio processing | librosa, soundfile |
| URL fetching | requests, BeautifulSoup, python-whois |
| Visualization | Matplotlib (Grad-CAM heatmaps) |
| Frontend | HTML, CSS, vanilla JavaScript |

## Project structure

```
DEtecT-it/
├── app.py                    # Flask routes: image/video/URL/audio analysis, report view/download
├── utils/
│   ├── model_loader.py       # Loads Xception + transforms
│   ├── face_extractor.py     # Face detection/extraction
│   ├── ela_analysis.py       # Error Level Analysis
│   ├── metadata_extractor.py # EXIF metadata forensics
│   ├── heatmap.py            # Grad-CAM generation
│   ├── video_analyzer.py     # Frame sampling for video analysis
│   ├── audio_analyzer.py     # AASIST-L voice spoof detection
│   ├── url_analyzer.py       # URL fetching + media extraction
│   └── report_generator.py   # Standalone HTML forensic report
├── templates/                 # Flask/Jinja2 HTML templates
├── static/                    # CSS, JS, uploads, generated results
├── models/                    # Pretrained checkpoints (not in git — see Setup)
├── tests/
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/anshikabisht111/DEtecT-it.git
cd DEtecT-it
pip install -r requirements.txt
```

Download the pretrained checkpoints (not committed to this repo — see `.gitignore`):

| File | Download | Place at |
|---|---|---|
| Xception (image/video) | [xception_best.pth](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth) | `models/xception_best.pth` |
| AASIST-L (audio) | [AASIST-L.pth](https://github.com/clovaai/aasist/blob/main/models/weights/AASIST-L.pth) | `models/AASIST-L.pth` |

```bash
python app.py
```
App runs at `http://localhost:5000`.

## Roadmap

- [x] URL-based analysis module
- [x] Video deepfake detection
- [x] Audio deepfake detection
- [ ] Dedicated AI-generated/synthetic image (GAN/diffusion) detection — active research area, not a quick add; see Scope & limitations

## Credits

- Image/video classifier weights: [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) (Yan et al., NeurIPS 2023), CC BY-NC 4.0
- Audio classifier weights: [AASIST](https://github.com/clovaai/aasist) (Jung et al., ICASSP 2022), MIT License

## Author

**Anshika Bisht** — MCA Cyber Security, Lovely Professional University
[GitHub](https://github.com/anshikabisht111)

---
*Built for academic/portfolio purposes. Not intended as a substitute for professional forensic analysis or legal evidence.*
