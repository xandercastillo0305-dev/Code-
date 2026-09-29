"""Preprocessing, hybrid model architecture and VERIFY_MEDIA."""
import numpy as np
import pytest
import torch

import config
from pipeline.preprocessing import FacePreprocessor, NoFaceDetectedError, NO_FACE_MESSAGE
from pipeline.verify import verify_media
from tests.conftest import NoFaceDetector, StubDetector


def test_preprocessing_outputs_380_normalised(face_images):
    r = FacePreprocessor(detector=StubDetector()).process_path(face_images[0])
    assert r.face.size == (380, 380)
    assert tuple(r.tensor.shape) == (3, 380, 380)
    assert r.tensor.min() < 0 < r.tensor.max()          # ImageNet normalisation applied


def test_no_face_raises_clear_error(face_images):
    with pytest.raises(NoFaceDetectedError) as exc:
        FacePreprocessor(detector=NoFaceDetector()).process_path(face_images[0])
    assert str(exc.value) == NO_FACE_MESSAGE == "No face detected. Case cannot be analyzed."


def test_low_probability_face_rejected(face_images):
    det = lambda im: (np.array([[0, 0, 50, 50]]), np.array([0.3]), None)
    with pytest.raises(NoFaceDetectedError):
        FacePreprocessor(detector=det).process_path(face_images[0])


def test_largest_face_selected():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 105, 105], [0, 0, 30, 30]])
    det = lambda im: (boxes, np.array([0.99, 0.99, 0.99]), None)
    box, _, _ = FacePreprocessor(detector=det).detect_largest_face(None)
    assert list(box) == [5, 5, 105, 105]


def test_verify_media_returns_case_fields(face_images, stub_bundle):
    out = verify_media(*face_images, tau=0.7, preprocessor=FacePreprocessor(detector=StubDetector()),
                       model_bundle=stub_bundle)
    for key in ("ai_classification", "confidence_score", "cosine_similarity", "euclidean_similarity",
                "ssim", "aggregated_score", "threshold", "model_version", "model_trained"):
        assert key in out
    assert out["ai_classification"] in ("Real", "Deepfake")
    assert (out["aggregated_score"] >= 0.7) == (out["ai_classification"] == "Real")
    assert out["suspect_face"].size == (380, 380)
    assert out["model_trained"] is False


def test_verify_media_identical_images_is_real(face_images, stub_bundle):
    out = verify_media(face_images[0], face_images[0], tau=0.7,
                       preprocessor=FacePreprocessor(detector=StubDetector()), model_bundle=stub_bundle)
    assert out["aggregated_score"] == pytest.approx(1.0)
    assert out["ai_classification"] == "Real"


@pytest.mark.slow
def test_real_hybrid_architecture_shapes():
    """Builds the actual EfficientNet-B4 || ViT-S/16 model (random init, no download)."""
    from pipeline.model import HybridCNNTransformer
    m = HybridCNNTransformer(pretrained=False).eval()
    assert m.cnn_dim == 1792
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        f1, f2 = m.branch_features(x)
        emb, logit = m(x)
    assert f1.shape == (2, 1792) and f2.shape == (2, 384)
    assert emb.shape == (2, config.EMBEDDING_DIM) and logit.shape == (2,)
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-5)


def test_load_model_falls_back_and_flags_untrained(monkeypatch, tmp_path):
    from pipeline import model as model_mod

    class Dummy(torch.nn.Module):
        def __init__(self, **kw):
            super().__init__()
    monkeypatch.setattr(model_mod, "HybridCNNTransformer", lambda pretrained=True, **kw: Dummy())
    b = model_mod.load_model(str(tmp_path / "missing.pt"), device="cpu")
    assert b.model_trained is False and "untrained" in b.model_version
