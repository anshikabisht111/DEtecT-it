from pathlib import Path
from datetime import datetime


def generate_report(image_filename, verdict, confidence, ela_results, metadata_results, heatmap_path=None, ela_image_path=None, output_path=None, confidence_tier=None, signal_conflict=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    case_id = datetime.now().strftime("DT-%Y%m%d-%H%M%S")
    verdict_color = "#2ecc71" if verdict == "REAL" else ("#f0c040" if verdict == "INCONCLUSIVE" else "#e74c3c")

    # Tier note: the standalone report can be read without ever seeing the
    # web results page, so it needs to carry the same "how sure is the
    # model, really" caveat that the page shows — otherwise a low-confidence
    # verdict reads as a confident one.
    if verdict == "INCONCLUSIVE" or confidence_tier == "low":
        tier_note = '<p style="color:#f0c040;margin-top:.6rem;font-size:.85rem;">⚠ Model confidence too low for a reliable verdict — manual review recommended.</p>'
    elif confidence_tier == "moderate":
        tier_note = '<p style="color:#e0c070;margin-top:.6rem;font-size:.85rem;">Moderate confidence — corroborate with the ELA and metadata findings below before relying on this verdict alone.</p>'
    elif confidence_tier == "high":
        tier_note = '<p style="color:#888;margin-top:.6rem;font-size:.85rem;">High confidence verdict.</p>'
    else:
        tier_note = ""

    conflict_html = ""
    if signal_conflict:
        conflict_html = f'<p style="color:#f0a0a0;margin-top:.6rem;font-size:.85rem;">🔍 <strong>Signals disagree:</strong> {signal_conflict} Recommend manual review before treating this as authentic.</p>'

    anomalies_html = ""
    for a in metadata_results.get("anomalies", []):
        anomalies_html += f'<li style="color:#f39c12;padding:4px 0;">⚠ {a}</li>'
    if not anomalies_html:
        anomalies_html = '<li style="color:#2ecc71;">✓ No anomalies detected.</li>'

    ela_img_html = f'<img src="../static/results/{Path(ela_image_path).name}" style="width:100%;border-radius:6px;"/>' if ela_image_path and Path(ela_image_path).exists() else '<p style="color:#888;">Not available.</p>'
    heatmap_img_html = f'<img src="../static/results/{Path(heatmap_path).name}" style="width:100%;border-radius:6px;"/>' if heatmap_path and Path(heatmap_path).exists() else '<p style="color:#888;">Not available.</p>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<title>DEtecT-it Report — {case_id}</title>
<style>
body{{background:#0f0f1a;color:#e0e0f0;font-family:'Segoe UI',sans-serif;padding:2rem;}}
.card{{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem;}}
h1{{color:#6c63ff;}} h2{{color:#6c63ff;font-size:.9rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:1rem;}}
.verdict{{font-size:2rem;font-weight:800;color:{verdict_color};}}
.bar{{background:#2a2a4a;border-radius:4px;height:10px;margin-top:6px;}}
.bar-fill{{height:100%;background:{verdict_color};border-radius:4px;width:{confidence}%;}}
table{{width:100%;border-collapse:collapse;font-size:.85rem;}}
td{{padding:.4rem .6rem;border-bottom:1px solid #2a2a4a;color:#8888aa;}}
td:first-child{{color:#e0e0f0;}}
.footer{{text-align:center;color:#555;font-size:.75rem;margin-top:2rem;}}
</style></head><body>
<h1>DEtecT-it — Forensic Report</h1>
<div class="card">
  <h2>Verdict</h2>
  <div class="verdict">{verdict}</div>
  <p style="color:#888;margin-top:.3rem;">Confidence: {confidence}%</p>
  <div class="bar"><div class="bar-fill"></div></div>
  {tier_note}
  {conflict_html}
  <p style="color:#888;margin-top:.8rem;">File: {image_filename} | Case: {case_id} | {timestamp}</p>
</div>
<div class="card">
  <h2>ELA Analysis</h2>
  <p style="color:#888;font-size:.85rem;">Mean Error: {ela_results.get('mean_error','N/A')} | Anomaly Score: {ela_results.get('anomaly_score','N/A')} | Suspicious: {'Yes ⚠' if ela_results.get('suspicious') else 'No ✓'}</p>
  {ela_img_html}
</div>
<div class="card">
  <h2>Grad-CAM Heatmap</h2>
  <p style="color:#888;font-size:.85rem;">Regions the model weighted most heavily for its verdict.</p>
  {heatmap_img_html}
</div>
<div class="card">
  <h2>Metadata Analysis</h2>
  <p style="color:#888;font-size:.85rem;">{metadata_results.get('summary','')}</p>
  <ul style="list-style:none;padding:0;">{anomalies_html}</ul>
</div>
<div class="card">
  <h2>Methodology</h2>
  <p style="color:#888;font-size:.85rem;line-height:1.7;">Xception backbone, using pretrained weights from DeepfakeBench (Yan et al., NeurIPS 2023), trained on all four FaceForensics++ manipulation methods (DeepFakes, Face2Face, FaceSwap, NeuralTextures; c23 compression). CC BY-NC 4.0 licensed, used here for non-commercial academic purposes. Error Level Analysis for compression artifact detection. EXIF metadata forensics for authenticity verification. Grad-CAM visualizes which regions influenced the CNN's output. Known limitation: trained on the FaceForensics++ family of manipulation methods — deepfakes made with unrelated/newer generation tools may not be reliably detected, which reflects an open research problem (cross-dataset generalization) in this field, not a defect specific to this implementation.</p>
</div>
<div class="footer">DEtecT-it v1.0 | Anshika | MCA Cyber Security, LPU | For investigative use only.</div>
</body></html>"""

    out = Path(output_path) if output_path else Path(f"static/results/{case_id}_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)