import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from src.config import BASE_DIR, LEVEL_DIRS, CLASSES, CNN_INPUT_SIZE, TEST_SIZE, RANDOM_SEED


class DataLoader:
    def __init__(self):
        self.classes = CLASSES

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and resize a single image to CNN_INPUT_SIZE, returns float32 [0,1] (H,W,3)."""
        img = Image.open(path).convert("RGB")
        img = img.resize(CNN_INPUT_SIZE, Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    def load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            images : (N, H, W, 3) float32
            labels : (N,) str  — one of CLASSES values
        """
        images, labels = [], []

        for folder, label in LEVEL_DIRS.items():
            folder_path = BASE_DIR / folder
            img_paths = sorted(folder_path.glob("*.jpg"))
            print(f"  {label:12s} ({folder}) — {len(img_paths)} images")

            for p in img_paths:
                try:
                    images.append(self._load_image(p))
                    labels.append(label)
                except Exception as e:
                    print(f"    [WARN] Skipping {p.name}: {e}")

        return np.array(images, dtype=np.float32), np.array(labels)

    def split(
        self, images: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified 80/20 train-test split."""
        return train_test_split(
            images,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
