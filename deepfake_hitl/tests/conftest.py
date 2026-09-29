"""Shared fixtures. Heavy components (MTCNN, the hybrid model) are replaced by
light stubs so the suite runs quickly on CPU / CI."""
import os
import tempfile

# Redirect all data paths BEFORE config is imported anywhere.
_TMP = tempfile.mkdtemp(prefix="dfhitl-test-")
os.environ.setdefault("DFHITL_DATA_DIR", os.path.join(_TMP, "data"))
os.environ.setdefault("DFHITL_WEIGHTS_DIR", os.path.join(_TMP, "weights"))

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pipeline.model import ModelBundle


def make_face_image(path, seed=0, size=256, variant=0):
    """Synthetic 'face-like' test image (no real person)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (200, 180, 160))
    d = ImageDraw.Draw(img)
    d.ellipse((48, 32, 208, 224), fill=(230, 190, 160))
    d.ellipse((90, 100, 115, 115), fill=(40, 40, 40))
    d.ellipse((141, 100, 166, 115), fill=(40, 40, 40))
    d.line((128, 115, 120, 155), fill=(150, 100, 90), width=3)
    d.arc((95, 160, 161, 195), 20, 160, fill=(150, 40, 40), width=4)
    arr = np.asarray(img).astype(int)
    arr += rng.normal(0, 4 + 30 * variant, arr.shape).astype(int)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)
    return path


class StubDetector:
    """Mimics MTCNN.detect(img, landmarks=True): one centred face."""

    def __call__(self, img):
        w, h = img.size
        box = np.array([[w * 0.2, h * 0.15, w * 0.8, h * 0.85]])
        lms = np.array([[[w * 0.4, h * 0.42], [w * 0.6, h * 0.42], [w * 0.5, h * 0.55],
                         [w * 0.42, h * 0.7], [w * 0.58, h * 0.7]]])
        return box, np.array([0.999]), lms


class NoFaceDetector:
    def __call__(self, img):
        return None, [None], None


class TinyHybrid(nn.Module):
    """Same interface as HybridCNNTransformer, but tiny and deterministic."""

    def __init__(self, dim=32):
        super().__init__()
        torch.manual_seed(0)
        self.cnn = nn.Sequential(nn.Conv2d(3, 8, 5, stride=4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.vit = nn.Sequential(nn.Conv2d(3, 8, 16, stride=16), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.projection = nn.Sequential(nn.Linear(16, dim), nn.LayerNorm(dim))
        self.head = nn.Linear(dim, 1)

    def branch_features(self, x):
        return self.cnn(x), self.vit(F.interpolate(x, size=(384, 384)))

    def fuse(self, f1, f2):
        return self.projection(torch.cat([f1, f2], 1))

    def forward(self, x):
        z = self.fuse(*self.branch_features(x))
        return F.normalize(z, dim=1), self.head(z).squeeze(1)

    def embed(self, x):
        return self.forward(x)[0]


@pytest.fixture
def stub_bundle():
    return ModelBundle(TinyHybrid().eval(), "cpu", "test-stub-untrained", False, "random-init")


@pytest.fixture
def face_images(tmp_path):
    a = make_face_image(tmp_path / "suspect.png", seed=1)
    b = make_face_image(tmp_path / "reference.png", seed=2)
    return str(a), str(b)
