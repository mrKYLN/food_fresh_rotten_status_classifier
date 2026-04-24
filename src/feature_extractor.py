"""
MobileNetV2 feature extractor (frozen ImageNet weights).
Produces 1280-dim embedding per image via GlobalAveragePooling.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from src.config import CNN_BATCH_SIZE


class MobileNetV2Extractor:
    def __init__(self):
        self.device = self._get_device()
        self.model = self._build_model().to(self.device)
        self.model.eval()

        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _build_model() -> nn.Module:
        weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1
        base = tv_models.mobilenet_v2(weights=weights)
        # Keep feature layers, add global avg pool + flatten
        model = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        for param in model.parameters():
            param.requires_grad = False
        return model

    def extract(self, images_np: np.ndarray) -> np.ndarray:
        """
        Args:
            images_np: (N, H, W, 3) float32 in [0, 1]
        Returns:
            features: (N, 1280) float32
        """
        # (N, H, W, 3) → (N, 3, H, W)
        tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2)
        all_features = []

        with torch.no_grad():
            for start in range(0, len(tensor), CNN_BATCH_SIZE):
                batch = tensor[start : start + CNN_BATCH_SIZE].to(self.device)
                batch = self.normalize(batch)
                feats = self.model(batch)
                all_features.append(feats.cpu().numpy())

        features = np.vstack(all_features).astype(np.float32)

        # Guard against NaN/Inf (can occur with MPS float32 numerical instability)
        bad = ~np.isfinite(features).all(axis=1)
        if bad.any():
            print(f"  [WARN] {bad.sum()} feature rows had NaN/Inf — clamping to 0")
            features[bad] = 0.0

        return features

    def extract_single(self, image_np: np.ndarray) -> np.ndarray:
        """image_np: (H, W, 3) float32 → (1280,) float32"""
        return self.extract(image_np[np.newaxis])[0]
