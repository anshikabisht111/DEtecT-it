const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const filePreview = document.getElementById("file-preview");
const fileName = document.getElementById("file-name");
const uploadForm = document.getElementById("upload-form");
const loadingOverlay = document.getElementById("loading-overlay");
const progressFill = document.getElementById("progress-fill");
const progressText = document.getElementById("progress-text");

const steps = [
  { id: "step-extract", label: "Extracting faces..." },
  { id: "step-model", label: "Running AI model..." },
  { id: "step-ela", label: "ELA analysis..." },
  { id: "step-meta", label: "Metadata forensics..." },
  { id: "step-report", label: "Generating report..." },
];

if (dropZone) {
  ["dragenter", "dragover"].forEach(e => {
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add("drag-over"); });
  });
  ["dragleave", "drop"].forEach(e => {
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove("drag-over"); });
  });
  dropZone.addEventListener("drop", ev => {
    const files = ev.dataTransfer.files;
    if (files.length > 0) { fileInput.files = files; showPreview(files[0]); }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) showPreview(fileInput.files[0]);
  });
}

function showPreview(file) {
  if (fileName) fileName.textContent = file.name;
  if (filePreview) filePreview.classList.add("show");
  const imgPreviewWrap = document.getElementById("img-preview-wrap");
  const previewImg = document.getElementById("preview-img");
  if (imgPreviewWrap && previewImg && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = e => { previewImg.src = e.target.result; imgPreviewWrap.classList.add("show"); };
    reader.readAsDataURL(file);
  }
}

if (uploadForm) {
  uploadForm.addEventListener("submit", () => {
    if (loadingOverlay) loadingOverlay.classList.add("show");
    runProgressAnimation();
  });
}

function runProgressAnimation() {
  let progress = 0;
  let stepIndex = 0;
  const interval = setInterval(() => {
    progress += 2;
    if (progressFill) progressFill.style.width = progress + "%";
    if (progressText) progressText.textContent = progress + "%";
    const newStep = Math.floor((progress / 100) * steps.length);
    if (newStep > stepIndex && stepIndex < steps.length) {
      const prevEl = document.getElementById(steps[stepIndex]?.id);
      if (prevEl) { prevEl.classList.remove("active"); prevEl.classList.add("done"); prevEl.querySelector(".step-icon").textContent = "✓"; prevEl.querySelector(".step-icon").classList.remove("spin"); }
      stepIndex = newStep;
      const currEl = document.getElementById(steps[stepIndex]?.id);
      if (currEl) { currEl.classList.add("active"); currEl.querySelector(".step-icon").textContent = "⟳"; currEl.querySelector(".step-icon").classList.add("spin"); }
    }
    if (progress >= 95) clearInterval(interval);
  }, 80);
}