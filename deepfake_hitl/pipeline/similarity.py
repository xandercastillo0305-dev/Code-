"""Multi-Metric Similarity Scoring (Section 3.3, Section 3.6, Appendix A).

All formulas are small pure functions so they can be unit-tested in
isolation (tests/test_similarity.py).

    cosine:     cos(A,B) = A.B / (||A|| ||B||)        -> mapped to [0,1] as (cos+1)/2
    euclidean:  d(A,B)   = sqrt(sum (A_i - B_i)^2)     on L2-normalised embeddings (0..2)
                euclidean_similarity = 1 - d/2         -> [0,1]
    SSIM:       skimage.metrics.structural_similarity  on the aligned grayscale face crops
    aggregate:  S = w_cos*cos + w_euc*euc + w_ssim*ssim, with w_cos + w_euc + w_ssim = 1
"""
import math

import numpy as np
from skimage.metrics import structural_similarity

EPS = 1e-12


def _as_vector(x):
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("Embedding vector is empty.")
    return v


def l2_normalize(x):
    """Return x / ||x||_2."""
    v = _as_vector(x)
    return v / max(np.linalg.norm(v), EPS)


def cosine_similarity(a, b):
    """Raw cosine similarity A.B / (||A|| ||B||), range [-1, 1]."""
    a, b = _as_vector(a), _as_vector(b)
    if a.shape != b.shape:
        raise ValueError("Embeddings must have the same dimension.")
    value = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))
    return max(-1.0, min(1.0, value))


def cosine_to_unit(cos):
    """Map a cosine similarity from [-1, 1] to [0, 1] with (cos + 1) / 2."""
    return (float(cos) + 1.0) / 2.0


def euclidean_distance(a, b):
    """d(A,B) = sqrt(sum_i (A_i - B_i)^2)."""
    a, b = _as_vector(a), _as_vector(b)
    if a.shape != b.shape:
        raise ValueError("Embeddings must have the same dimension.")
    return float(np.sqrt(np.sum((a - b) ** 2)))


def distance_to_similarity(d):
    """Convert a distance between unit vectors (range 0..2) to a similarity: 1 - d/2."""
    return max(0.0, min(1.0, 1.0 - float(d) / 2.0))


def euclidean_similarity(a, b):
    """euclidean_similarity = 1 - d(A,B)/2 computed on the L2-normalised embeddings."""
    return distance_to_similarity(euclidean_distance(l2_normalize(a), l2_normalize(b)))


def ssim_score(face_a, face_b):
    """SSIM between two aligned face crops.

    NOTE: SSIM is a *spatial* measure (it compares local luminance, contrast and
    structure in sliding windows), so it needs 2-D image data. It therefore
    cannot be computed on the 1-D fused embedding. Following the design, it
    is computed on the aligned grayscale 380x380 face crops of the suspect
    and reference images produced by preprocessing.py.

    Accepts PIL images or numpy arrays (H,W) / (H,W,3). The result is clipped
    to [0, 1] so it is on the same scale as the other two metrics (raw SSIM
    can be slightly negative for anti-correlated images).
    """
    ga, gb = _to_gray_uint8(face_a), _to_gray_uint8(face_b)
    if ga.shape != gb.shape:
        raise ValueError(f"Face crops must have the same size, got {ga.shape} and {gb.shape}.")
    value = structural_similarity(ga, gb, data_range=255)
    return max(0.0, min(1.0, float(value)))


def _to_gray_uint8(img):
    if hasattr(img, "convert"):          # PIL image
        return np.asarray(img.convert("L"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def validate_weights(weights):
    """Weights must be non-negative and sum to 1 (Appendix A)."""
    keys = {"cosine", "euclidean", "ssim"}
    if set(weights) != keys:
        raise ValueError(f"Metric weights must have exactly the keys {sorted(keys)}.")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Metric weights must be non-negative.")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Metric weights must sum to 1, got {total:.6f}.")
    return weights


def aggregate_score(cos, euc, ssim, weights):
    """S = w_cos*cos + w_euc*euc + w_ssim*ssim."""
    validate_weights(weights)
    return float(weights["cosine"] * cos + weights["euclidean"] * euc + weights["ssim"] * ssim)


def compute_similarity(emb_suspect, emb_reference, face_suspect, face_reference, weights):
    """Compute all three metrics and the aggregated score S. Returns a dict."""
    cos = cosine_to_unit(cosine_similarity(emb_suspect, emb_reference))
    euc = euclidean_similarity(emb_suspect, emb_reference)
    ssim = ssim_score(face_suspect, face_reference)
    return {
        "cosine_similarity": cos,
        "euclidean_similarity": euc,
        "ssim": ssim,
        "aggregated_score": aggregate_score(cos, euc, ssim, weights),
    }
