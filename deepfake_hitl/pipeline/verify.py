"""VERIFY_MEDIA(I, R, tau): the full algorithm of Section 3.6.

The step numbers in the comments (Step 2 ... Step 15) are the ones listed in
the README "Paper <-> Code mapping" table. Step 1 is the algorithm input
(the function signature).

    Step 1   Input: suspect image I, reference image R, threshold tau
    Step 2   Detect, crop and align the face in I            (MTCNN)
    Step 3   Detect, crop and align the face in R            (MTCNN)
    Step 4   Resize both faces to 380x380 and normalise (ImageNet mean/std)
    Step 5   F1_I, F1_R <- EfficientNet-B4(I'), EfficientNet-B4(R')    local features
    Step 6   F2_I, F2_R <- Transformer(I'),     Transformer(R')        global attention
    Step 7   F_I <- Fuse(F1_I, F2_I);  F_R <- Fuse(F1_R, F2_R)   (concat -> projection -> L2)
    Step 8   cos <- (CosineSimilarity(F_I, F_R) + 1) / 2
    Step 9   d <- EuclideanDistance(F_I, F_R);  euc <- 1 - d/2
    Step 10  ssim <- SSIM(face_I, face_R)
    Step 11  S <- w_cos*cos + w_euc*euc + w_ssim*ssim
    Step 12  if S >= tau then label <- "Real"
    Step 13  else label <- "Deepfake"
    Step 14  confidence <- Confidence(S, tau)
    Step 15  return (label, confidence, cos, euc, ssim, S) -> forwarded to Human-in-the-Loop review
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
    # Step 1: inputs
    tau = classifier.validate_threshold(config.THRESHOLD_TAU if tau is None else tau)
    weights = similarity.validate_weights(weights or config.METRIC_WEIGHTS)
    preprocessor = preprocessor or FacePreprocessor()
    if model_bundle is None:
        from pipeline.model import get_model_bundle
        model_bundle = get_model_bundle()

    # Step 2: detect/crop/align face in the suspect image I
    # Step 4 (for I): resize to 380x380 + ImageNet normalisation (inside process_path)
    suspect = preprocessor.process_path(suspect_path)
    # Step 3: detect/crop/align face in the reference image R
    # Step 4 (for R): resize to 380x380 + ImageNet normalisation
    reference = preprocessor.process_path(reference_path)

    batch = torch.stack([suspect.tensor, reference.tensor])
    model = model_bundle.model
    model.eval()
    with torch.no_grad():
        batch = batch.to(model_bundle.device)
        # Step 5: EfficientNet-B4 local features F1   |  run in parallel on
        # Step 6: Transformer global features F2      |  the same face
        f1, f2 = model.branch_features(batch)
        # Step 7: feature fusion -> 512-d L2-normalised embeddings F_I, F_R
        fused = F.normalize(model.fuse(f1, f2), p=2, dim=1).cpu().numpy()
    emb_i, emb_r = fused[0], fused[1]

    # Step 8: cosine similarity, mapped to [0, 1]
    cos = similarity.cosine_to_unit(similarity.cosine_similarity(emb_i, emb_r))
    # Step 9: Euclidean distance on the L2-normalised embeddings -> similarity
    euc = similarity.euclidean_similarity(emb_i, emb_r)
    # Step 10: SSIM on the aligned grayscale face crops (SSIM needs 2-D data)
    ssim = similarity.ssim_score(suspect.face, reference.face)
    # Step 11: weighted aggregation
    s = similarity.aggregate_score(cos, euc, ssim, weights)
    # Step 12-13: threshold rule (Real if S >= tau else Deepfake)
    label = classifier.classify(s, tau)
    # Step 14: confidence from the distance between S and tau
    conf = classifier.confidence(s, tau)

    # Step 15: return the preliminary result; the caller queues it for analyst review.
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
