import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in model.backbone.named_modules():
            target = module

        target.register_forward_hook(save_activation)
        target.register_full_backward_hook(save_gradient)

    def generate(self, input_tensor):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        score = logits[0, 1]

        self.model.zero_grad()
        score.backward()

        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activation = self.activations[0]

        for i, w in enumerate(pooled_grads):
            activation[i] *= w

        cam = activation.mean(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()

        return cam


def save_heatmap(original_image, cam, output_path, verdict=None, confidence=None):
    from PIL import Image as PILImage
    import matplotlib.cm as cm

    w, h = original_image.size
    cam_img = Image.fromarray(np.uint8(255 * cam)).resize((w, h), Image.LANCZOS)
    heatmap = np.uint8(cm.jet(np.array(cam_img) / 255.0)[:, :, :3] * 255)
    overlay = Image.blend(original_image.convert("RGB"), Image.fromarray(heatmap), alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    title = "Grad-CAM"
    if verdict and confidence:
        title += f"\n{verdict} ({confidence:.1f}%)"
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path