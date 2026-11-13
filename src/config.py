from pathlib import Path
import torch


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

OUTPUT_DIR = ROOT_DIR / "outputs"

BASELINE_DIR = OUTPUT_DIR / "baseline"
SKIP_DIR = OUTPUT_DIR / "skip_fusion"
PREDICTION_DIR = OUTPUT_DIR / "predictions"


# ==========================================
# REPRODUCIBILITY
# ==========================================

SEED = 42


# ==========================================
# IMAGE
# ==========================================

IMAGE_SIZE = 256


# ==========================================
# SEGMENTATION
# ==========================================

NUM_CLASSES = 2


# ==========================================
# TRAINING
# ==========================================

BATCH_SIZE = 8

EPOCHS = 15

LEARNING_RATE = 0.001

NUM_WORKERS = 0


# ==========================================
# DEVICE
# ==========================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("Using device:", DEVICE)