"""
utils/model_loader.py — REPLACES the previous EfficientNet-B4 version.

WHY THE SWITCH: the earlier checkpoint (effnb4_best.pth) was trained on only
ONE manipulation method (FF-NT / NeuralTextures) and failed badly on
real-world deepfakes made with other techniques. This checkpoint
(xception_best.pth) is trained on ALL FOUR FaceForensics++ manipulation
methods combined (DeepFakes, Face2Face, FaceSwap, NeuralTextures), which
generalizes noticeably better across different deepfake generation styles.

Still a real, honest limitation to state: this is still FaceForensics++
family data. A deepfake made with a completely unrelated tool/generator
(e.g. some phone app, or a diffusion-based face swap) may still be missed —
cross-dataset generalization is an open research problem in this field.
That's not a bug in this code, it's the current state of the art.

Source: DeepfakeBench (Yan et al., NeurIPS 2023) — https://github.com/SCLBD/DeepfakeBench
License: CC BY-NC 4.0 (non-commercial — fine for student portfolio use, cite it).
Checkpoint (direct download, no access form):
    https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth

Architecture code adapted from DeepfakeBench's training/networks/xception.py
(itself adapted from https://github.com/ondyari/FaceForensics), with their
internal `@BACKBONE.register_module` registry decorator removed since we're
using it standalone, outside their training framework.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

WEIGHTS_PATH = "models/xception_best.pth"
WEIGHTS_URL = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth"


# ---------------------------------------------------------------------------
# Xception architecture (verbatim from DeepfakeBench, registry decorator removed)
# ---------------------------------------------------------------------------

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    def __init__(self, xception_config):
        super().__init__()
        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        inc = xception_config["inc"]
        dropout = xception_config["dropout"]

        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        final_channel = 2048
        if self.mode == "adjust_channel_iid":
            final_channel = 512
            self.mode = "adjust_channel"
        self.last_linear = nn.Linear(final_channel, self.num_classes)
        if dropout:
            self.last_linear = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(final_channel, self.num_classes))

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def features(self, input):
        x = self.conv1(input); x = self.bn1(x); x = self.relu(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.block4(x); x = self.block5(x); x = self.block6(x); x = self.block7(x)
        x = self.block8(x); x = self.block9(x); x = self.block10(x); x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x); x = self.bn3(x); x = self.relu(x)
        x = self.conv4(x); x = self.bn4(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, features):
        if self.mode == "adjust_channel":
            x = features
        else:
            x = self.relu(features)
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(weights_path=WEIGHTS_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {"num_classes": 2, "mode": "original", "inc": 3, "dropout": False}
    model = Xception(config)

    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. Download it (no access form needed):\n"
            f"  {WEIGHTS_URL}\n"
            f"and place it at {weights_path} (create the models/ folder if needed)."
        )

    raw_state_dict = torch.load(weights_file, map_location=device)
    # checkpoint keys are prefixed "backbone." — strip it
    remapped = {k.replace("backbone.", "", 1): v for k, v in raw_state_dict.items()}

    model.load_state_dict(remapped, strict=True)
    print(f"[DEtecT-it] Loaded DeepfakeBench Xception checkpoint (all-4-methods) from {weights_path}")

    model.to(device)
    model.eval()
    return model, device


def get_transforms():
    """Same resolution/normalization convention as DeepfakeBench (256, mean/std=0.5)."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
