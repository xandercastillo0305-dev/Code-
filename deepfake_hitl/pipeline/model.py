"""Hybrid CNN-Transformer (Section 3.3, Algorithm 3.6 steps 4-6).

    face (3x380x380) --+--> EfficientNet-B4 (timm, pooled)   -> F1 (1792-d, local features)
                       |
                       +--> resize 384 -> ViT-S/16 (timm)    -> F2 (384-d CLS, global attention)

    F_fused = concat(F1, F2) -> Linear -> LayerNorm -> L2-normalise   (512-d embedding)

A binary head (Real/Deepfake) sits on the fused embedding. It is only used
during training (training/train.py) so the embedding learns manipulation-
sensitive features. At inference the system classifies with the similarity /
threshold method (similarity.py + classifier.py), exactly as in the paper.
"""
import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

log = logging.getLogger(__name__)


class HybridCNNTransformer(nn.Module):
    def __init__(self, cnn_name=config.CNN_BACKBONE, vit_name=config.VIT_BACKBONE,
                 embedding_dim=config.EMBEDDING_DIM, pretrained=True,
                 vit_input_size=config.VIT_INPUT_SIZE):
        super().__init__()
        import timm
        # CNN branch: EfficientNet-B4, classifier removed -> global-pooled 1792-d vector.
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0)
        # Transformer branch: ViT, classifier removed -> CLS-token embedding.
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0,
                                     img_size=vit_input_size)
        self.vit_input_size = vit_input_size
        self.cnn_dim = self.cnn.num_features       # 1792 for EfficientNet-B4
        self.vit_dim = self.vit.num_features       # 384 for ViT-Small
        # Feature fusion: concat(F1, F2) -> Linear -> LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(self.cnn_dim + self.vit_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        # Training-only classification head (logit of "Deepfake").
        self.head = nn.Linear(embedding_dim, 1)

    def branch_features(self, x):
        """Return (F1, F2) for a batch of preprocessed 380x380 faces."""
        f1 = self.cnn(x)
        # The ViT splits the image into 16x16 patches, so its input side must be
        # divisible by 16. 380 is not (380/16 = 23.75), so the 380x380 face is
        # resized to 384x384 *inside this branch only*; the CNN branch still
        # sees the native 380x380 EfficientNet-B4 input.
        x_vit = x
        if x.shape[-1] != self.vit_input_size or x.shape[-2] != self.vit_input_size:
            x_vit = F.interpolate(x, size=(self.vit_input_size, self.vit_input_size),
                                  mode="bilinear", align_corners=False)
        f2 = self.vit(x_vit)                        # pooled CLS token (timm global_pool="token")
        return f1, f2

    def fuse(self, f1, f2):
        """F_fused = concat(F1, F2) -> projection (pre-normalisation)."""
        return self.projection(torch.cat([f1, f2], dim=1))

    def forward(self, x):
        """Returns (L2-normalised embedding, deepfake logit)."""
        fused = self.fuse(*self.branch_features(x))
        return F.normalize(fused, p=2, dim=1), self.head(fused).squeeze(1)

    @torch.no_grad()
    def embed(self, x):
        return self.forward(x)[0]


@dataclass
class ModelBundle:
    model: HybridCNNTransformer
    device: str
    model_version: str
    model_trained: bool            # True only if fine-tuned weights were loaded
    backbone_source: str           # "fine-tuned", "imagenet" or "random-init"

    def embed(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model.embed(batch.to(self.device)).cpu().numpy()


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(pretrained=True, **kwargs):
    """Create the model; if ImageNet weights cannot be downloaded, fall back to random init."""
    if pretrained:
        try:
            return HybridCNNTransformer(pretrained=True, **kwargs), "imagenet"
        except Exception as exc:  # no network / HF blocked
            log.warning("Could not load ImageNet weights (%s); using random init.", exc)
    return HybridCNNTransformer(pretrained=False, **kwargs), "random-init"


def load_model(weights_path=config.HYBRID_WEIGHTS_PATH, device=None):
    """Load fine-tuned weights if present, else ImageNet backbones (UNTRAINED banner)."""
    device = device or default_device()
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        arch = ckpt.get("arch", {})
        model = HybridCNNTransformer(
            cnn_name=arch.get("cnn_name", config.CNN_BACKBONE),
            vit_name=arch.get("vit_name", config.VIT_BACKBONE),
            embedding_dim=arch.get("embedding_dim", config.EMBEDDING_DIM),
            vit_input_size=arch.get("vit_input_size", config.VIT_INPUT_SIZE),
            pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        version = ckpt.get("model_version", config.MODEL_VERSION)
        bundle = ModelBundle(model, device, version, True, "fine-tuned")
    else:
        model, source = build_model(pretrained=True)
        bundle = ModelBundle(model, device, f"{config.MODEL_VERSION}-untrained-{source}",
                             False, source)
    bundle.model.to(device).eval()
    return bundle


_cached = {}


def get_model_bundle(weights_path=config.HYBRID_WEIGHTS_PATH, device=None):
    """Process-wide cache so the web app loads the model once."""
    key = (weights_path, device)
    if key not in _cached:
        _cached[key] = load_model(weights_path, device)
    return _cached[key]


def weights_available(weights_path=config.HYBRID_WEIGHTS_PATH):
    return bool(weights_path) and os.path.exists(weights_path)
