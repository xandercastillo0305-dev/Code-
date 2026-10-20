"""VERIFY_MEDIA(I, R, tau): the Algorithm of Section 3.6.

Comments marked "Line N" use the line numbers of the manuscript's algorithm
(Section 3.6, revised version with 21 lines). Lines 2-16 run in this
function; Line 17 (queue) is in app.py::submit_case, and Lines 18-20 run in
app.py::review_case after an analyst acts.

    Line 1   procedure VERIFY_MEDIA(I, R, tau)
    Line 2     for each X in {I, R} do                     suspect and reference processed identically
    Line 3       X_face <- MTCNN_Detect_Largest_Face(X)
    Line 4       if X_face is empty: return "No face detected. Case cannot be analyzed."
    Line 5       X_aligned <- Align_Crop_Resize(X, X_face, 380x380)
    Line 6       X_norm <- Normalize(X_aligned)            ImageNet mean/std
    Line 7       F_1 <- EfficientNet_B4(X_norm)            1,792-d local features
    Line 8       F_2 <- Vision_Transformer(Resize(X_norm, 384x384))   384-d global features
    Line 9       X_fused <- L2_Normalize(LayerNorm(Linear(Concat(F_1, F_2))))   512-d
    Line 10    end for
    Line 11    Score_Cos  <- (CosineSimilarity(I_fused, R_fused) + 1) / 2
    Line 12    Score_Euc  <- 1 - EuclideanDistance(I_fused, R_fused) / 2
    Line 13    Score_SSIM <- SSIM(Grayscale(I_aligned), Grayscale(R_aligned))
    Line 14    S <- w_cos*Score_Cos + w_euc*Score_Euc + w_ssim*Score_SSIM
    Line 15    if S >= tau then C <- "Real" else C <- "Deepfake"
    Line 16    conf <- 0.5 + 0.5 * min(1, |S - tau| / max(tau, 1 - tau))
    Line 17    add case to the review queue as "pending"           app.py::submit_case
    Line 18    D, Rationale <- Human_In_The_Loop_Review(...)       app.py::review_case -> review/workflow.py
    Line 19    FR <- Generate_Forensic_Report(...)                 reports/report_generator.py::generate_pdf
    Line 20    return C, S, D, FR                                  stored in the case record
    Line 21  end procedure

Implementation notes:
* The "for each X in {I, R}" loop (Lines 2-10) is run as one batch, which gives
  the same result as processing the images one after the other.
* A case may contain several suspect images (I_1..I_k) and several reference
  images (R_1..R_m). Lines 11-16 are evaluated for every suspect against every
  reference; each suspect keeps its best-matching reference (highest S), and
  the case is classified by its most suspicious suspect image (lowest S).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import config
from pipeline import classifier, similarity
from pipeline.preprocessing import NO_FACE_MESSAGE, FacePreprocessor, NoFaceDetectedError


def verify_media(suspect_path, reference_path, tau=None, *, preprocessor=None,
                 model_bundle=None, weights=None):
    """One suspect image and one reference image (the original case type).

    Returns a dict with every AI field of the case record (Appendix C) plus
    the aligned face crops (``suspect_face`` / ``reference_face``, PIL).
    Raises preprocessing.NoFaceDetectedError if either image has no face.
    """
    out = verify_case([suspect_path], [reference_path], tau, preprocessor=preprocessor,
                      model_bundle=model_bundle, weights=weights)
    out["suspect_face"] = out["suspect_faces"][0]
    out["reference_face"] = out["reference_faces"][0]
    return out


def _preprocess_all(preprocessor, paths, label):
    results = []
    for n, path in enumerate(paths, start=1):
        try:
            results.append(preprocessor.process_path(path))
        except NoFaceDetectedError:
            # Line 4: no face -> stop; name the image so the investigator can fix it
            raise NoFaceDetectedError(
                f"{NO_FACE_MESSAGE} ({label} image {n}: {os.path.basename(path)})")
    return results


def verify_case(suspect_paths, reference_paths, tau=None, *, preprocessor=None,
                model_bundle=None, weights=None):
    """Run the automated analysis for a case with one or more images of each kind.

    Every suspect image is compared with every reference image; for each
    suspect the best-matching reference (highest S) is kept, since an
    authentic image should match at least one genuine photo of the subject.
    The case is classified Deepfake if ANY suspect image falls below tau, and
    the case-level scores are those of the most suspicious image (lowest S).
    """
    if not suspect_paths or not reference_paths:
        raise ValueError("At least one suspect image and one reference image are required.")
    # Line 1: procedure VERIFY_MEDIA(I, R, tau)
    tau = classifier.validate_threshold(config.THRESHOLD_TAU if tau is None else tau)
    weights = similarity.validate_weights(weights or config.METRIC_WEIGHTS)
    preprocessor = preprocessor or FacePreprocessor()
    if model_bundle is None:
        from pipeline.model import get_model_bundle
        model_bundle = get_model_bundle()

    # Line 2: for each X in {I, R}   (here: every suspect and every reference image)
    # Lines 3-6: MTCNN face detection, no-face stop, align/crop/resize 380x380, normalise
    suspects = _preprocess_all(preprocessor, suspect_paths, "suspect")
    references = _preprocess_all(preprocessor, reference_paths, "reference")

    batch = torch.stack([x.tensor for x in suspects + references])
    model = model_bundle.model
    model.eval()
    with torch.no_grad():
        batch = batch.to(model_bundle.device)
        # Line 7: EfficientNet-B4 local features F_1          |  both branches run on the
        # Line 8: ViT global features F_2 (resized to 384x384) |  same faces
        f1, f2 = model.branch_features(batch)
        # Line 9: fusion -> 512-d L2-normalised embeddings   (Line 10: end for)
        fused = F.normalize(model.fuse(f1, f2), p=2, dim=1).cpu().numpy()
    emb_s, emb_r = fused[:len(suspects)], fused[len(suspects):]

    per_suspect = []
    for i, sus in enumerate(suspects):
        best = None
        for j, ref in enumerate(references):
            # Line 11: cosine similarity, mapped to [0, 1]
            cos = similarity.cosine_to_unit(similarity.cosine_similarity(emb_s[i], emb_r[j]))
            # Line 12: Euclidean distance on the L2-normalised embeddings -> 1 - d/2
            euc = similarity.euclidean_similarity(emb_s[i], emb_r[j])
            # Line 13: SSIM on the aligned grayscale face crops
            ssim = similarity.ssim_score(sus.face, ref.face)
            # Line 14: aggregated score S
            s = similarity.aggregate_score(cos, euc, ssim, weights)
            if best is None or s > best["aggregated_score"]:
                best = {"best_reference": j + 1, "cosine_similarity": float(cos),
                        "euclidean_similarity": float(euc), "ssim": float(ssim),
                        "aggregated_score": float(s)}
        # Line 15: if S >= tau then "Real" else "Deepfake";  Line 16: confidence
        best["n"] = i + 1
        best["ai_classification"] = classifier.classify(best["aggregated_score"], tau)
        best["confidence_score"] = float(classifier.confidence(best["aggregated_score"], tau))
        per_suspect.append(best)

    # Case level: the most suspicious suspect image (lowest S) decides.
    deciding = min(per_suspect, key=lambda r: r["aggregated_score"])

    # Preliminary result -> app.py adds it to the queue as "pending" (Line 17);
    # Lines 18-20 run in app.py::review_case once an analyst reviews it.
    return {
        "ai_classification": deciding["ai_classification"],
        "confidence_score": deciding["confidence_score"],
        "cosine_similarity": deciding["cosine_similarity"],
        "euclidean_similarity": deciding["euclidean_similarity"],
        "ssim": deciding["ssim"],
        "aggregated_score": deciding["aggregated_score"],
        "threshold": tau,
        "model_version": model_bundle.model_version,
        "model_trained": bool(model_bundle.model_trained),
        "suspect_results": per_suspect,
        "deciding_suspect": deciding["n"],
        "suspect_faces": [x.face for x in suspects],
        "reference_faces": [x.face for x in references],
        "embedding_dim": int(np.asarray(emb_s[0]).shape[0]),
    }
