"""Evaluate the hybrid model and the XceptionNet baseline (Section 3.7, Chapter 4).

1. Frame-level detection on data/test/{real,fake}: both models' classification
   heads -> accuracy, precision, recall, F1, ROC-AUC (Deepfake = positive).
2. If data/test/pairs.csv exists: the full paper pipeline (similarity score S
   and threshold tau) on suspect/reference pairs.

Outputs (evaluation/results/): comparison table (.csv + .md), confusion-matrix
PNGs and one ROC-curve PNG with all methods.

    python -m evaluation.evaluate_model --data-root data
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
from evaluation.metrics import classification_metrics  # noqa: E402
from pipeline.model import default_device, load_model  # noqa: E402
from training.dataset import FaceCropDataset, eval_transform, read_pairs  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@torch.no_grad()
def head_scores(model, dataset, device, batch_size=16):
    model.eval()
    scores, labels = [], []
    for x, y in DataLoader(dataset, batch_size):
        scores.append(torch.sigmoid(model(x.to(device))[1]).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels).astype(int)


def summarize(name, y, score, pred):
    m = classification_metrics(y.tolist(), pred.tolist(), score.tolist())
    m["method"] = name
    return m


def save_confusion(m, out_dir):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ConfusionMatrixDisplay(np.array(m["confusion_matrix"]), display_labels=["Real", "Deepfake"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(m["method"], fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, f"confusion_{m['method'].lower().replace(' ', '_').replace('/', '-')}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_roc(curves, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for name, y, s in curves:
        if len(set(y.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        m = classification_metrics(y.tolist(), (s >= 0.5).astype(int).tolist(), s.tolist())
        ax.plot(fpr, tpr, label=f"{name} (AUC {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — Deepfake = positive")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, "roc_curve.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_table(rows, out_dir):
    cols = ["method", "n", "accuracy", "precision", "recall", "f1", "roc_auc"]
    fmt = lambda v: "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([[fmt(r[c]) for c in cols] for r in rows])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(out_dir, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(config.BASE_DIR, "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--hybrid-weights", default=config.HYBRID_WEIGHTS_PATH)
    ap.add_argument("--baseline-weights", default=config.BASELINE_WEIGHTS_PATH)
    ap.add_argument("--tau", type=float, default=config.THRESHOLD_TAU)
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = args.device or default_device()
    os.makedirs(args.out, exist_ok=True)

    rows, curves = [], []
    bundle = load_model(args.hybrid_weights, device)
    if not bundle.model_trained:
        print("WARNING: hybrid model is UNTRAINED (no fine-tuned weights) - results are not meaningful.")
    ds = FaceCropDataset(args.data_root, args.split, eval_transform(config.IMAGE_SIZE))
    s, y = head_scores(bundle.model, ds, device)
    rows.append(summarize("Hybrid CNN-Transformer (head)", y, s, (s >= 0.5).astype(int)))
    curves.append(("Hybrid (head)", y, s))

    if os.path.exists(args.baseline_weights):
        from training.train_baseline import XCEPTION_SIZE, load_baseline
        base = load_baseline(args.baseline_weights, device)
        ds_b = FaceCropDataset(args.data_root, args.split, eval_transform(XCEPTION_SIZE))
        sb, yb = head_scores(base, ds_b, device)
        rows.append(summarize("XceptionNet baseline", yb, sb, (sb >= 0.5).astype(int)))
        curves.append(("XceptionNet", yb, sb))
    else:
        print(f"Baseline weights not found at {args.baseline_weights}; run training/train_baseline.py first.")

    pairs_csv = os.path.join(args.data_root, args.split, "pairs.csv")
    if os.path.exists(pairs_csv):
        from pipeline.calibrate_threshold import pair_scores
        S, yp = pair_scores(read_pairs(pairs_csv), bundle)
        fake_score = 1.0 - S                        # low similarity => Deepfake
        rows.append(summarize(f"Hybrid similarity S (tau={args.tau:.2f})", yp, fake_score,
                              (S < args.tau).astype(int)))
        curves.append(("Hybrid similarity (1 - S)", yp, fake_score))

    write_table(rows, args.out)
    for r in rows:
        save_confusion(r, args.out)
    save_roc(curves, args.out)
    print(f"Saved results to {args.out}")
    return rows


if __name__ == "__main__":
    main()
