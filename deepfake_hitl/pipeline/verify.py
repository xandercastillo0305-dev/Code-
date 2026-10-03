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

Implementation note: the "for each X in {I, R}" loop (Lines 2-10) is run as
one batch of two images, which gives the same result as processing them one
after the other.
"""
import numpy as np
import torch
import torch.nn.functional as F

import config
from pipeline import classifier, similarity
from pipeline.preprocessing import FacePreprocessor


def verify_media(suspect_path, reference_path, tau=None, *, preprocessor=None,
                 model_bundle=None, weights=None):
    """Run the automated analysis for one case.

    Returns a dict with every AI field of the case record (Appendix C) plus
    the aligned face crops (``suspect_face`` / ``reference_face``, PIL) so the
    caller can store them for the redacted review view.
    Raises preprocessing.NoFaceDetectedError if either image has no face.
    """
    # Line 1: procedure VERIFY_MEDIA(I, R, tau)
    tau = classifier.validate_threshold(config.THRESHOLD_TAU if tau is None else tau)
    weights = similarity.validate_weights(weights or config.METRIC_WEIGHTS)
    preprocessor = preprocessor or FacePreprocessor()
    if model_bundle is None:
        from pipeline.model import get_model_bundle
        model_bundle = get_model_bundle()

    # Line 2: for each X in {I, R}
    # Line 3: MTCNN detects the largest face
    # Line 4: no face -> NoFaceDetectedError ("No face detected. Case cannot be analyzed.")
    # Line 5: align (eyes level), crop with margin, resize to 380x380
    # Line 6: ImageNet normalisation          (Lines 3-6 are inside process_path)
    suspect = preprocessor.process_path(suspect_path)       # X = I
    reference = preprocessor.process_path(reference_path)   # X = R

    batch = torch.stack([suspect.tensor, reference.tensor])
    model = model_bundle.model
    model.eval()
    with torch.no_grad():
        batch = batch.to(model_bundle.device)
        # Line 7: EfficientNet-B4 local features F_1          |  both branches run on the
        # Line 8: ViT global features F_2 (resized to 384x384) |  same faces (I and R)
        f1, f2 = model.branch_features(batch)
        # Line 9: fusion -> 512-d L2-normalised embeddings I_fused, R_fused   (Line 10: end for)
        fused = F.normalize(model.fuse(f1, f2), p=2, dim=1).cpu().numpy()
    emb_i, emb_r = fused[0], fused[1]

    # Line 11: cosine similarity, mapped to [0, 1]
    cos = similarity.cosine_to_unit(similarity.cosine_similarity(emb_i, emb_r))
    # Line 12: Euclidean distance on the L2-normalised embeddings -> similarity 1 - d/2
    euc = similarity.euclidean_similarity(emb_i, emb_r)
    # Line 13: SSIM on the aligned grayscale face crops (SSIM needs 2-D data)
    ssim = similarity.ssim_score(suspect.face, reference.face)
    # Line 14: aggregated score S (weighted sum)
    s = similarity.aggregate_score(cos, euc, ssim, weights)
    # Line 15: if S >= tau then "Real" else "Deepfake"
    label = classifier.classify(s, tau)
    # Line 16: confidence score
    conf = classifier.confidence(s, tau)

    # Preliminary result -> app.py adds it to the queue as "pending" (Line 17);
    # Lines 18-20 run in app.py::review_case once an analyst reviews it.
    return {
        "ai_classification": label,
        "confidence_score": float(conf),
        "cosine_similarity": float(cos),
        "euclidean_similarity": float(euc),
        "ssim": float(ssim),
        "aggregated_score": float(s),
        "threshold": tau,
        "model_version": model_bundle.model_version,
        "model_trained": bool(model_bundle.model_trained),
        "suspect_face": suspect.face,
        "reference_face": reference.face,
        "embedding_dim": int(np.asarray(emb_i).shape[0]),
    }
