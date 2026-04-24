from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Dataset
LEVEL_DIRS = {
    "level_1": "Taze",      # Fresh
    "level_2": "Yenebilir", # Edible (borderline)
    "level_3": "Çürük",     # Rotten
}
CLASSES = list(LEVEL_DIRS.values())

# Directories
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "feature_cache").mkdir(exist_ok=True)

# CNN feature extractor
CNN_INPUT_SIZE = (224, 224)
CNN_BATCH_SIZE = 32

# SOM
SOM_X = 3
SOM_Y = 1
PCA_COMPONENTS = 64
SOM_SIGMA = 0.5
SOM_LEARNING_RATE = 0.5
SOM_ITERATIONS = 500

# Training
TEST_SIZE = 0.20
RANDOM_SEED = 42
