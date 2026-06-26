from pathlib import Path
import os
import random

import numpy as np
import torch


PROJECT_ROOT = Path.cwd()
RAW_DATA_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR
TORCH_HOME = PROJECT_ROOT / ".torch_cache"

CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
BASELINE_EPOCHS = 3
RESNET_EPOCHS = 5
LEARNING_RATE = 1e-3
ALLOW_DOWNLOAD_WEIGHTS = True


def setup_environment(seed: int = SEED):
    OUTPUT_DIR.mkdir(exist_ok=True)
    TORCH_HOME.mkdir(exist_ok=True)
    os.environ["TORCH_HOME"] = str(TORCH_HOME)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = setup_environment()

