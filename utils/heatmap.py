"""
utils/heatmap.py — Grad-CAM visualization.

Target layer is `bn4` — the last conv-based feature layer before pooling in
the DeepfakeBench Xception architecture (model_loader.py). Note: this
model's forward() returns a TUPLE (logits, features), not just logits —
GradCAM.generate() unpacks that accordingly.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None

        def save_activation(module, input, output):
            # NOTE: do NOT detach here — we need this tensor to stay attached
            # to the autograd graph so we can compute gradients w.r.t. it
            # directly via torch.autograd.grad(). A full_backward_hook was
            # tried first but conflicts with this model's inplace ReLU
            # (shared `self.relu` module used elsewhere in forward) —
            # PyTorch raises "BackwardHookFunction output modified inplace".
            # Using autograd.grad() sidesteps that entirely.
            self.activations = output

        target = model.bn4
        target.register_forward_hook(save_activation)

    def generate(self, input_tensor, class_idx=1):
        """class_idx=1 -> fake class (real=0, fake=1 convention)."""
        self.model.eval()

        logits, _ = self.model(input_tensor)  # this model returns (logits, features)
        score = logits[0, class_idx]

        grads = torch.autograd.grad(score, self.activations, retain_graph=False)[0]
        activations = self.activations.detach()[0]  # (C, H, W)
        gradients = grads.detach()[0]                # (C, H, W)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


def save_heatmap(original_image, cam, output_path, verdict="", confidence=None):
    """
    original_image: PIL Image (pre-normalization, as originally loaded)
    cam: 2D numpy array from GradCAM.generate(), values in [0,1]
    """
    orig_np = np.array(original_image.convert("RGB"))
    h, w = orig_np.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.5 * orig_np + 0.5 * heatmap).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    title = "Grad-CAM"
    if verdict:
        title += f"\n{verdict}" + (f" ({confidence}%)" if confidence is not None else "")
    axes[1].imshow(overlay)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
