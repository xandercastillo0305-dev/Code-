"""Central configuration for the Deepfake Detection HITL prototype.

Every tunable value from Chapter 3 lives here so the panel can see it in one
place: the decision threshold tau, the similarity-metric weights, the input
image size, model names and all data paths.

Paths can be redirected with the environment variable ``DFHITL_DATA_DIR``
(used by the test-suite so tests never touch the real case store).
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Paths (Section 3.9). Uploads live OUTSIDE static/ and are only served by
# authenticated routes (Appendix D).
# --------------------------------------------------------------------------
DATA_DIR = os.environ.get("DFHITL_DATA_DIR", os.path.join(BASE_DIR, "data"))
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
CASES_PATH = os.path.join(DATA_DIR, "cases.json")
USERS_PATH = os.path.join(DATA_DIR, "users.json")
AUDIT_LOG_PATH = os.path.join(DATA_DIR, "audit_log.jsonl")

WEIGHTS_DIR = os.environ.get("DFHITL_WEIGHTS_DIR", os.path.join(BASE_DIR, "weights"))
HYBRID_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "hybrid_best.pt")
BASELINE_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "xception_best.pt")

# --------------------------------------------------------------------------
# Preprocessing (Section 3.3 / 3.6)
# --------------------------------------------------------------------------
IMAGE_SIZE = 380                      # EfficientNet-B4 native resolution
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FACE_MARGIN = 0.20                    # extra border around the MTCNN box (fraction of box size)
MIN_FACE_PROBABILITY = 0.90           # MTCNN detections below this are treated as "no face"

# --------------------------------------------------------------------------
# Hybrid CNN-Transformer (Section 3.3)
# --------------------------------------------------------------------------
CNN_BACKBONE = "efficientnet_b4"
VIT_BACKBONE = "vit_small_patch16_384"
VIT_INPUT_SIZE = 384                  # patch-16 ViT needs a size divisible by 16
EMBEDDING_DIM = 512
MODEL_VERSION = "hybrid-effb4-vits16-v1.0"

# --------------------------------------------------------------------------
# Multi-metric similarity + classification (Section 3.6, Appendix A)
# --------------------------------------------------------------------------
THRESHOLD_TAU = 0.70
METRIC_WEIGHTS = {
    "cosine": 1 / 3,
    "euclidean": 1 / 3,
    "ssim": 1 / 3,
}

# --------------------------------------------------------------------------
# Human-in-the-loop workflow (Section 3.4)
# --------------------------------------------------------------------------
MIN_RATIONALE_LENGTH = 30             # characters; enforced on server and client
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_MB = 10                    # per image
MAX_IMAGE_PIXELS = 40_000_000         # decompression-bomb guard

# Flask
SECRET_KEY = os.environ.get("DFHITL_SECRET_KEY", "dev-only-change-me")
