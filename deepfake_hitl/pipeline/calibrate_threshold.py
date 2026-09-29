"""Calibrate the decision threshold tau on a labelled validation set (Section 3.6).

For every validation pair the aggregated score S is computed with the live
pipeline. tau is then chosen to either
  * maximise F1 (Deepfake = positive class; prediction Deepfake if S < tau), or
  * maximise Youden's J = TPR - FPR on the ROC curve.

    python -m pipeline.calibrate_threshold --pairs data/val/pairs.csv --method f1
    python -m pipeline.calibrate_threshold --scores-csv val_scores.csv --method youden

Copy the printed value into config.THRESHOLD_TAU.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import f1_score, roc_curve  # noqa: E402

import config  # noqa: E402
from pipeline import similarity  # noqa: E402


def pair_scores(pairs, model_bundle, weights=None, preprocessor=None):
    """Aggregated score S for each (suspect, reference, label) pair.

    If ``preprocessor`` is None the images are assumed to be aligned face crops
    already (output of training/extract_faces.py) and are only resized /
    normalised; otherwise MTCNN preprocessing is run like in the live system.
    """
    import torch
    from pipeline.preprocessing import to_tensor
    weights = weights or config.METRIC_WEIGHTS
    scores, labels = [], []
    for s_path, r_path, label in pairs:
        faces = []
        for p in (s_path, r_path):
            if preprocessor is not None:
                faces.append(preprocessor.process_path(p).face)
            else:
                with Image.open(p) as im:
                    faces.append(im.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
        emb = model_bundle.embed(torch.stack([to_tensor(f) for f in faces]))
        scores.append(similarity.compute_similarity(emb[0], emb[1], faces[0], faces[1], weights)["aggregated_score"])
        labels.append(label)
    return np.asarray(scores), np.asarray(labels)


def best_threshold_f1(scores, labels):
    """tau maximising F1 for 'Deepfake if S < tau' (labels: 1 = Deepfake)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    uniq = np.unique(s)
    candidates = np.concatenate([[uniq[0]], (uniq[:-1] + uniq[1:]) / 2, [uniq[-1] + 1e-6]])
    candidates = candidates[(candidates > 0) & (candidates < 1)]
    best_tau, best_f1 = 0.5, -1.0
    for tau in candidates:
        f1 = f1_score(y, (s < tau).astype(int), zero_division=0)
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), float(f1)
    return best_tau, best_f1


def best_threshold_youden(scores, labels):
    """tau maximising Youden's J = TPR - FPR (Deepfake score = 1 - S)."""
    s = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(np.asarray(labels, dtype=int), 1.0 - s)
    finite = np.isfinite(thr)
    j = tpr[finite] - fpr[finite]
    k = int(np.argmax(j))
    # positive if (1 - S) >= t  <=>  S <= 1 - t ; nudge so the rule "S < tau" matches
    tau = float(np.clip(1.0 - thr[finite][k] + 1e-6, 1e-6, 1 - 1e-6))
    return tau, float(j[k])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs", help="pairs.csv (suspect,reference,label)")
    src.add_argument("--scores-csv", help="precomputed csv with columns score,label (real|fake)")
    ap.add_argument("--method", choices=["f1", "youden"], default="f1")
    ap.add_argument("--run-mtcnn", action="store_true", help="images are raw frames, not face crops")
    ap.add_argument("--save-scores", default=None, help="write the computed scores to this csv")
    args = ap.parse_args(argv)

    if args.scores_csv:
        with open(args.scores_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        scores = np.array([float(r["score"]) for r in rows])
        labels = np.array([1 if r["label"].strip().lower() in ("fake", "deepfake", "1") else 0 for r in rows])
    else:
        from pipeline.model import load_model
        from pipeline.preprocessing import FacePreprocessor
        from training.dataset import read_pairs
        bundle = load_model()
        if not bundle.model_trained:
            print("WARNING: no fine-tuned weights found; calibrating an UNTRAINED model.")
        scores, labels = pair_scores(read_pairs(args.pairs), bundle,
                                     preprocessor=FacePreprocessor() if args.run_mtcnn else None)
        if args.save_scores:
            with open(args.save_scores, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["score", "label"])
                w.writerows((f"{s:.6f}", "fake" if l else "real") for s, l in zip(scores, labels))

    if args.method == "f1":
        tau, value = best_threshold_f1(scores, labels)
        print(f"Best tau (max F1) = {tau:.4f}   F1 = {value:.4f}   (n={len(scores)})")
    else:
        tau, value = best_threshold_youden(scores, labels)
        print(f"Best tau (Youden's J) = {tau:.4f}   J = {value:.4f}   (n={len(scores)})")
    print(f"Set THRESHOLD_TAU = {tau:.2f} in config.py")
    return tau


if __name__ == "__main__":
    main()
