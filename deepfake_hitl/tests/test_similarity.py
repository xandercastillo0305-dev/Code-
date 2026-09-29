"""Appendix A similarity formulas."""
import math

import numpy as np
import pytest

import config
from pipeline import similarity as sim


def test_identical_vectors_have_perfect_similarity():
    v = np.random.default_rng(0).normal(size=512)
    assert sim.cosine_similarity(v, v) == pytest.approx(1.0)
    assert sim.cosine_to_unit(sim.cosine_similarity(v, v)) == pytest.approx(1.0)
    assert sim.euclidean_distance(v, v) == pytest.approx(0.0)
    assert sim.euclidean_similarity(v, v) == pytest.approx(1.0)


def test_cosine_formula_known_values():
    assert sim.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert sim.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert sim.cosine_to_unit(-1.0) == 0.0
    assert sim.cosine_to_unit(0.0) == 0.5
    assert sim.cosine_similarity([1, 2, 3], [4, 5, 6]) == pytest.approx(32 / (math.sqrt(14) * math.sqrt(77)))


def test_euclidean_formula_and_range():
    assert sim.euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)
    # opposite unit vectors are at the maximum distance 2 -> similarity 0
    assert sim.euclidean_similarity([1, 0], [-1, 0]) == pytest.approx(0.0)
    # orthogonal unit vectors: d = sqrt(2)
    assert sim.euclidean_similarity([1, 0], [0, 1]) == pytest.approx(1 - math.sqrt(2) / 2)


def test_similarity_decreases_monotonically_with_distance():
    rng = np.random.default_rng(1)
    base = sim.l2_normalize(rng.normal(size=128))
    direction = sim.l2_normalize(rng.normal(size=128))
    cos_vals, euc_vals = [], []
    for step in np.linspace(0, 3, 15):
        other = sim.l2_normalize(base + step * direction)
        cos_vals.append(sim.cosine_similarity(base, other))
        euc_vals.append(sim.euclidean_similarity(base, other))
    assert all(a >= b - 1e-12 for a, b in zip(cos_vals, cos_vals[1:]))
    assert all(a >= b - 1e-12 for a, b in zip(euc_vals, euc_vals[1:]))
    distances = np.linspace(0, 2, 21)
    sims = [sim.distance_to_similarity(d) for d in distances]
    assert all(a > b for a, b in zip(sims, sims[1:]))


def test_ssim_identical_images_is_one_and_noise_lowers_it():
    rng = np.random.default_rng(2)
    img = (rng.random((380, 380)) * 255).astype(np.uint8)
    assert sim.ssim_score(img, img) == pytest.approx(1.0)
    noisy = np.clip(img.astype(int) + rng.normal(0, 40, img.shape), 0, 255).astype(np.uint8)
    very_noisy = np.clip(img.astype(int) + rng.normal(0, 90, img.shape), 0, 255).astype(np.uint8)
    s1, s2 = sim.ssim_score(img, noisy), sim.ssim_score(img, very_noisy)
    assert 0.0 <= s2 < s1 < 1.0


def test_ssim_accepts_pil_rgb():
    from PIL import Image
    arr = (np.random.default_rng(3).random((64, 64, 3)) * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    assert sim.ssim_score(im, im) == pytest.approx(1.0)


def test_default_weights_sum_to_one():
    assert sum(config.METRIC_WEIGHTS.values()) == pytest.approx(1.0)
    sim.validate_weights(config.METRIC_WEIGHTS)


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        sim.validate_weights({"cosine": 0.5, "euclidean": 0.5, "ssim": 0.5})
    with pytest.raises(ValueError):
        sim.validate_weights({"cosine": 1.0, "euclidean": 0.0})
    with pytest.raises(ValueError):
        sim.aggregate_score(1, 1, 1, {"cosine": 1.2, "euclidean": -0.2, "ssim": 0.0})


def test_aggregate_is_weighted_sum():
    w = {"cosine": 0.5, "euclidean": 0.3, "ssim": 0.2}
    assert sim.aggregate_score(0.9, 0.8, 0.5, w) == pytest.approx(0.45 + 0.24 + 0.10)
    assert sim.aggregate_score(1, 1, 1, config.METRIC_WEIGHTS) == pytest.approx(1.0)
