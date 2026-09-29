"""Smoke tests: training, checkpoint loading, calibration, evaluation scripts."""
import csv
import os

import numpy as np
import pytest

from evaluation.evaluate_agreement import agreement_summary
from pipeline.calibrate_threshold import best_threshold_f1, best_threshold_youden
from tests.conftest import make_face_image

SMALL = ["--cnn", "efficientnet_b0", "--vit", "vit_tiny_patch16_224", "--image-size", "64",
         "--vit-size", "64", "--no-pretrained", "--device", "cpu", "--num-workers", "0"]


@pytest.fixture
def dataset(tmp_path):
    for split in ("train", "val", "test"):
        for cls, variant in (("real", 0), ("fake", 1)):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(3):
                make_face_image(d / f"id{i}_{cls}_{i}.png", seed=i + 10 * variant, size=96, variant=variant)
    return tmp_path


def test_calibration_f1_finds_separating_tau():
    s = np.array([0.9, 0.85, 0.8, 0.4, 0.35, 0.3])
    y = np.array([0, 0, 0, 1, 1, 1])
    tau, f1 = best_threshold_f1(s, y)
    assert 0.4 < tau <= 0.8 and f1 == 1.0
    tau_j, j = best_threshold_youden(s, y)
    assert 0.4 < tau_j <= 0.8 + 1e-5 and j == pytest.approx(1.0)
    assert ((s < tau_j).astype(int) == y).all()


def test_calibration_cli_scores_csv(tmp_path, capsys):
    from pipeline.calibrate_threshold import main
    p = tmp_path / "scores.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "label"])
        w.writerows([(0.9, "real"), (0.8, "real"), (0.5, "fake"), (0.45, "fake")])
    tau = main(["--scores-csv", str(p), "--method", "f1"])
    assert 0.5 < tau <= 0.8 and "THRESHOLD_TAU" in capsys.readouterr().out


def test_train_saves_checkpoint_that_loads_as_trained(dataset, tmp_path):
    from pipeline.model import load_model
    from training.train import main
    out = tmp_path / "w" / "hybrid_best.pt"
    main(["--data-root", str(dataset), "--epochs", "1", "--batch-size", "2", "--max-steps", "1",
          "--out", str(out)] + SMALL)
    assert out.exists()
    bundle = load_model(str(out), device="cpu")
    assert bundle.model_trained is True and bundle.backbone_source == "fine-tuned"


def test_evaluate_model_writes_results(dataset, tmp_path):
    from evaluation.evaluate_model import main as evaluate
    from training.train import main as train
    w = tmp_path / "hybrid.pt"
    train(["--data-root", str(dataset), "--epochs", "1", "--batch-size", "2", "--max-steps", "1",
           "--out", str(w)] + SMALL)
    with open(dataset / "test" / "pairs.csv", "w", newline="") as f:
        csv.writer(f).writerows([["suspect", "reference", "label"],
                                 ["real/id0_real_0.png", "real/id1_real_1.png", "real"],
                                 ["fake/id0_fake_0.png", "real/id1_real_1.png", "fake"]])
    out = tmp_path / "results"
    rows = evaluate(["--data-root", str(dataset), "--hybrid-weights", str(w), "--baseline-weights",
                     str(tmp_path / "none.pt"), "--out", str(out), "--device", "cpu"])
    assert len(rows) == 2
    assert (out / "roc_curve.png").exists() and (out / "comparison.md").exists()
    assert any(p.name.startswith("confusion_") for p in out.iterdir())


def test_build_pairs(dataset):
    from training.build_pairs import build_pairs
    path = build_pairs(str(dataset / "val"))
    rows = list(csv.DictReader(open(path)))
    assert rows and all(r["reference"].startswith("real/") and r["suspect"] != r["reference"] for r in rows)


def test_agreement_summary():
    def case(ai, dec, final, status):
        return {"status": status, "ai_classification": ai, "review_decision": dec, "final_classification": final}
    cases = [case("Real", "Confirm", "Real", "verified"), case("Deepfake", "Confirm", "Deepfake", "verified"),
             case("Deepfake", "Override", "Real", "verified"), case("Real", "Flag", "Inconclusive", "flagged"),
             case("Real", None, None, "pending")]
    s = agreement_summary(cases)
    assert (s["confirm"], s["override"], s["flag"], s["pending"]) == (2, 1, 1, 1)
    assert s["agreement_rate"] == pytest.approx(2 / 3)
    assert s["strict_agreement_rate"] == pytest.approx(2 / 4)
    assert s["cohens_kappa"] == pytest.approx(s["cohens_kappa_manual"])
    assert agreement_summary([])["agreement_rate"] is None
