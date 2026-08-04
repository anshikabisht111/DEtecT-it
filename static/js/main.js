// ── Scanning animation ──────────────────────────────────────────
function createScanOverlay() {
  const overlay = document.createElement('div');
  overlay.id = 'scan-overlay';
  overlay.innerHTML = `
    <div class="scan-frame">
      <div class="scan-corner tl"></div>
      <div class="scan-corner tr"></div>
      <div class="scan-corner bl"></div>
      <div class="scan-corner br"></div>
      <div class="scan-line"></div>
      <div class="scan-text">ANALYZING...</div>
    </div>
  `;
  overlay.style.cssText = `
    display:none;position:fixed;inset:0;background:rgba(8,11,18,0.95);
    z-index:9999;align-items:center;justify-content:center;flex-direction:column;
    backdrop-filter:blur(10px);
  `;
  document.body.appendChild(overlay);
  return overlay;
}

// ── Drag & Drop ─────────────────────────────────────────────────
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const fileName   = document.getElementById('file-name');
const uploadForm = document.getElementById('upload-form');

if (dropZone) {
  ['dragenter','dragover'].forEach(e => {
    dropZone.addEventListener(e, ev => {
      ev.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });
  ['dragleave','drop'].forEach(e => {
    dropZone.addEventListener(e, ev => {
      ev.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });
  dropZone.addEventListener('drop', ev => {
    const files = ev.dataTransfer.files;
    if (files.length > 0) { fileInput.files = files; showPreview(files[0]); }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) showPreview(fileInput.files[0]);
  });
}

function showPreview(file) {
  if (fileName) fileName.textContent = file.name;
  if (filePreview) filePreview.classList.add('show');
  const wrap = document.getElementById('img-preview-wrap');
  const img  = document.getElementById('preview-img');
  if (wrap && img && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = e => {
      img.src = e.target.result;
      wrap.classList.add('show');
      startScanOnPreview(img);
    };
    reader.readAsDataURL(file);
  }
}

// ── Scan effect on preview image ─────────────────────────────────
function startScanOnPreview(imgEl) {
  const existing = document.getElementById('img-scan-wrap');
  if (existing) existing.replaceWith(imgEl);

  const wrap = document.createElement('div');
  wrap.id = 'img-scan-wrap';
  wrap.style.cssText = 'position:relative;display:inline-block;margin-top:.5rem;';

  const scanLine = document.createElement('div');
  scanLine.style.cssText = `
    position:absolute;left:0;right:0;top:0;height:2px;
    background:linear-gradient(90deg,transparent,#38bdf8,#a855f7,transparent);
    z-index:10;animation:imgScan 2s ease-in-out 3;box-shadow:0 0 10px #38bdf8;
  `;

  if (!document.getElementById('img-scan-style')) {
    const st = document.createElement('style');
    st.id = 'img-scan-style';
    st.textContent = `@keyframes imgScan{0%{top:0;opacity:1}100%{top:100%;opacity:0}}`;
    document.head.appendChild(st);
  }

  imgEl.parentNode && imgEl.parentNode.replaceChild(wrap, imgEl);
  wrap.appendChild(imgEl);
  wrap.appendChild(scanLine);
}

// ── Loading overlay with progress ────────────────────────────────
const steps = [
  { id: 'step-extract', label: 'Extracting faces...' },
  { id: 'step-model',   label: 'Running AI model...' },
  { id: 'step-ela',     label: 'ELA analysis...' },
  { id: 'step-meta',    label: 'Metadata forensics...' },
  { id: 'step-report',  label: 'Generating report...' },
];

if (uploadForm) {
  uploadForm.addEventListener('submit', () => {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) { overlay.style.display = 'flex'; runProgress(); }
  });
}

function runProgress() {
  let progress = 0, stepIndex = 0;
  const fill = document.getElementById('progress-fill');
  const text = document.getElementById('progress-text');

  // Activate first step
  const first = document.getElementById(steps[0].id);
  if (first) { first.classList.add('active'); first.querySelector('.step-icon').textContent = '⟳'; first.querySelector('.step-icon').classList.add('spin'); }

  const interval = setInterval(() => {
    progress += 1.5;
    if (fill) fill.style.width = progress + '%';
    if (text) text.textContent = Math.round(progress) + '%';

    const newStep = Math.min(Math.floor((progress / 100) * steps.length), steps.length - 1);
    if (newStep > stepIndex) {
      const prev = document.getElementById(steps[stepIndex].id);
      if (prev) { prev.classList.remove('active'); prev.classList.add('done'); prev.querySelector('.step-icon').textContent = '✓'; prev.querySelector('.step-icon').classList.remove('spin'); }
      stepIndex = newStep;
      const curr = document.getElementById(steps[stepIndex].id);
      if (curr) { curr.classList.add('active'); curr.querySelector('.step-icon').textContent = '⟳'; curr.querySelector('.step-icon').classList.add('spin'); }
    }
    if (progress >= 95) clearInterval(interval);
  }, 60);
}

// ── Counter animation on result page ────────────────────────────
function animateCounters() {
  const confidence = document.getElementById('confidence-val');
  const faces      = document.getElementById('faces-val');

  if (confidence) {
    const target = parseFloat(confidence.dataset.target);
    let current = 0;
    const step = target / 60;
    const interval = setInterval(() => {
      current = Math.min(current + step, target);
      confidence.textContent = current.toFixed(1) + '%';
      if (current >= target) clearInterval(interval);
    }, 16);
  }

  if (faces) {
    const target = parseInt(faces.dataset.target);
    let current = 0;
    const interval = setInterval(() => {
      current = Math.min(current + 1, target);
      faces.textContent = current;
      if (current >= target) clearInterval(interval);
    }, 80);
  }
}

// ── Circular confidence gauge ────────────────────────────────────
function drawGauge(confidence, verdict) {
  const canvas = document.getElementById('confidence-gauge');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2 + 20;
  const radius = 80;
  const startAngle = Math.PI * 0.75;
  const endAngle   = Math.PI * 2.25;
  const color = verdict === 'REAL' ? '#34d399' : (verdict === 'INCONCLUSIVE' ? '#f0d060' : '#f87171');
  let current = 0;

  function draw(val) {
    ctx.clearRect(0, 0, W, H);
    // Track
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.stroke();
    // Fill
    const fillEnd = startAngle + (val / 100) * (endAngle - startAngle);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, fillEnd);
    ctx.strokeStyle = color;
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;
    ctx.stroke();
    // Text
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 28px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(Math.round(val) + '%', cx, cy);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('CONFIDENCE', cx, cy + 22);
  }

  const interval = setInterval(() => {
    current = Math.min(current + confidence / 60, confidence);
    draw(current);
    if (current >= confidence) clearInterval(interval);
  }, 16);
}

// ── Init on result page ──────────────────────────────────────────
window.addEventListener('load', () => {
  animateCounters();
  const gauge = document.getElementById('confidence-gauge');
  if (gauge) {
    const conf    = parseFloat(gauge.dataset.confidence);
    const verdict = gauge.dataset.verdict;
    drawGauge(conf, verdict);
  }
});
