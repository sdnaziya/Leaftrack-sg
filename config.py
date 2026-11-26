"""
LeafTrack-SG Configuration
Central configuration for all paths, model settings, and constants.
"""
import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
UPLOADS_DIR     = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
STATIC_DIR      = os.path.join(BASE_DIR, "static")

# ─── Dataset Paths ────────────────────────────────────────────────────────────
PLANTVILLAGE_DIR    = os.path.join(DATA_DIR, "plantvillage")
PLANTDOC_DIR        = os.path.join(DATA_DIR, "plantdoc")
SEGMENTATION_DIR    = os.path.join(DATA_DIR, "segmentation")
SEG_IMAGES_DIR      = os.path.join(SEGMENTATION_DIR, "images")
SEG_MASKS_DIR       = os.path.join(SEGMENTATION_DIR, "masks")

# ─── Model Paths ──────────────────────────────────────────────────────────────
CLASSIFIER_DIR      = os.path.join(MODELS_DIR, "classifier")
SEGMENTATION_MODEL_DIR = os.path.join(MODELS_DIR, "segmentation")

CLASSIFIER_MODEL_PATH   = os.path.join(CLASSIFIER_DIR, "efficientnet_classifier.keras")
SEGMENTATION_MODEL_PATH = os.path.join(SEGMENTATION_MODEL_DIR, "unet_model.keras")
CLASS_NAMES_PATH        = os.path.join(CLASSIFIER_DIR, "class_names.json")
LABEL_MAP_PATH          = os.path.join(CLASSIFIER_DIR, "label_map.json")

# ─── Image Settings ───────────────────────────────────────────────────────────
IMAGE_SIZE          = (224, 224)        # EfficientNetB0 input
SEG_IMAGE_SIZE      = (256, 256)        # U-Net input
MAX_IMAGE_SIZE_MB   = 10
ALLOWED_EXTENSIONS  = {"jpg", "jpeg", "png", "webp", "bmp"}

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE          = 32
EPOCHS              = 40          # more epochs for custom CNN from scratch
LEARNING_RATE       = 1e-3        # 1e-3 is correct for custom CNN; 1e-4 causes collapse
FINE_TUNE_LR        = 1e-4        # used for fine-tuning stage
VALIDATION_SPLIT    = 0.15
TEST_SPLIT          = 0.10

# U-Net training
SEG_BATCH_SIZE      = 8
SEG_EPOCHS          = 40

# ─── Severity Thresholds ──────────────────────────────────────────────────────
SEVERITY_LOW_MAX        = 20.0   # < 20% infected → Low
SEVERITY_MODERATE_MAX   = 50.0   # 20-50% infected → Moderate
# > 50% → High

# ─── Database ─────────────────────────────────────────────────────────────────
DATABASE_PATH   = os.path.join(BASE_DIR, "database", "leaftrack.db")

# ─── Flask ────────────────────────────────────────────────────────────────────
SECRET_KEY          = "leaftrack-sg-secret-2024"
DEBUG               = False
HOST                = "0.0.0.0"
PORT                = 5000

# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
GRADCAM_LAYER_NAME  = "top_conv"   # Last conv layer of EfficientNetB0
GRADCAM_ALPHA       = 0.4          # Overlay transparency

# ─── Auto-create required directories ─────────────────────────────────────────
for _dir in [UPLOADS_DIR, REPORTS_DIR, CLASSIFIER_DIR, SEGMENTATION_MODEL_DIR]:
    os.makedirs(_dir, exist_ok=True)
