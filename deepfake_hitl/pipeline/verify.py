"""VERIFY_MEDIA(I, R, tau): the Algorithm of Section 3.6.

Comments marked "Line N" use the line numbers of the manuscript's algorithm
(thesis slide "Algorithm Overview"). Lines 2-15 (the automated part) are in
this function; lines 16-18 run later in the web app, after an analyst acts.

    Line 1   procedure VERIFY_MEDIA(I, R, tau)
    Line 2   I_cropped, I_aligned <- MTCNN(I)                 face detection + alignment
    Line 3   I_norm <- Resize_And_Normalize(I_cropped, 380x380)
    Line 4   F_1 <- EfficientNet-B4(I_norm)                    local texture features
    Line 5   F_2 <- Vision Transformer(I_norm)                 global attention
    Line 6   F_fused <- Concatenate(F_1, F_2)                  (+ Linear -> LayerNorm -> L2)
    Line 7   Score_Cos  <- CosineSimilarity(F_fused, R)        mapped to [0,1] as (cos+1)/2
    Line 8   Score_Euc  <- EuclideanDistance(F_fused, R)       converted to 1 - d/2
    Line 9   Score_SSIM <- SSIM(face crop of I, face crop of R)
    Line 10  S <- w_cos*Score_Cos + w_euc*Score_Euc + w_ssim*Score_SSIM
    Line 11-15  if S >= tau then C <- "Real" else C <- "Deepfake"
    Line 16  D, Rationale <- Human_In_The_Loop_Review(...)    app.py::review_case -> review/workflow.py
    Line 17  FR <- Generate_Forensic_Report(...)              reports/report_generator.py::generate_pdf
    Line 18  return D, S, FR

Notes on how the code realises the pseudocode:
* R is the reference *image*; lines 2-6 are applied to it as well, so R in
  lines 7-8 is the reference's fused embedding.
* Line 9: SSIM needs 2-D image data, so it compares the aligned face crops
  (not the 1-D embeddings).
* Line 8: the distance is converted to a similarity (1 - d/2) so that, like
  the other two metrics, higher means "more similar" before aggregation.
* A confidence score (classifier.confidence) is computed next to lines 11-15;
  it is not a separate line in the pseudocode.
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

    # Line 2: MTCNN face detection, cropping and alignment for I
    # Line 3: resize to 380x380 + ImageNet normalisation (inside process_path)
    suspect = preprocessor.process_path(suspect_path)
    # Lines 2-3 applied to the reference image R
    reference = preprocessor.process_path(reference_path)

    batch = torch.stack([suspect.tensor, reference.tensor])
    model = model_bundle.model
    model.eval()
    with torch.no_grad():
        batch = batch.to(model_bundle.device)
        # Line 4: EfficientNet-B4 local features F_1   |  run in parallel on
        # Line 5: Transformer global features F_2      |  the same face (I and R)
        f1, f2 = model.branch_features(batch)
        # Line 6: feature fusion -> 512-d L2-normalised embeddings (F_fused for I and R)
        fused = F.normalize(model.fuse(f1, f2), p=2, dim=1).cpu().numpy()
    emb_i, emb_r = fused[0], fused[1]

    # Line 7: cosine similarity, mapped to [0, 1]
    cos = similarity.cosine_to_unit(similarity.cosine_similarity(emb_i, emb_r))
    # Line 8: Euclidean distance on the L2-normalised embeddings -> similarity 1 - d/2
    euc = similarity.euclidean_similarity(emb_i, emb_r)
    # Line 9: SSIM on the aligned grayscale face crops (SSIM needs 2-D data)
    ssim = similarity.ssim_score(suspect.face, reference.face)
    # Line 10: aggregated score S (weighted sum)
    s = similarity.aggregate_score(cos, euc, ssim, weights)
    # Lines 11-15: if S >= tau then "Real" else "Deepfake"
    label = classifier.classify(s, tau)
    # Confidence score (computed alongside lines 11-15)
    conf = classifier.confidence(s, tau)

    # Preliminary result -> queued as "pending" for Line 16 (Human-in-the-Loop review)
    # and Line 17 (forensic report), which run in app.py once an analyst reviews it.
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
