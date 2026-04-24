"""
Hybrid CNN-SOM-SVM sınıflandırıcı.

MobileNetV2 → StandardScaler → PCA(64) → MiniSom(3×1) [topoloji] → SVM-RBF [tahmin]
Tahminler SVM ile yapılır; SOM yalnızca görselleştirme içindir.
float64: PCA randomized SVD'de float32 overflow'unu önler.
"""
import warnings
import numpy as np
import joblib
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from src.config import (
    SOM_X, SOM_Y, PCA_COMPONENTS,
    SOM_SIGMA, SOM_LEARNING_RATE, SOM_ITERATIONS,
    MODELS_DIR, RANDOM_SEED, CLASSES,
)


class HybridCNNSOM:
    def __init__(
        self,
        som_x: int = SOM_X,
        som_y: int = SOM_Y,
        n_components: int = PCA_COMPONENTS,
        sigma: float = SOM_SIGMA,
        learning_rate: float = SOM_LEARNING_RATE,
        num_iterations: int = SOM_ITERATIONS,
        svm_C: float = 10.0,
        svm_gamma: str = "scale",
    ):
        self.som_x = som_x
        self.som_y = som_y
        self.n_components = n_components
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.svm_C = svm_C
        self.svm_gamma = svm_gamma

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        self.som = None
        self.svm = SVC(
            kernel="rbf",
            C=svm_C,
            gamma=svm_gamma,
            probability=True,
            random_state=RANDOM_SEED,
        )

        self.winner_label_map = {}
        self._raw_label_map = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "HybridCNNSOM":
        np.random.seed(RANDOM_SEED)

        features_pca = self._fit_preprocess(features)

        # ---- SOM (topology / visualisation) --------------------------
        self.som = MiniSom(
            x=self.som_x,
            y=self.som_y,
            input_len=self.n_components,
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            random_seed=RANDOM_SEED,
        )
        self._init_som_centroids(features_pca, labels)
        print(f"  Training SOM ({self.som_x}×{self.som_y}, "
              f"{self.num_iterations} iterations)...")
        self.som.train_random(features_pca, self.num_iterations, verbose=True)
        self._build_label_map(features_pca, labels)

        # ---- SVM (classification) ------------------------------------
        print("  Fitting SVM classifier (rbf kernel)...")
        self.svm.fit(features_pca, labels)
        print("  SVM fitted.")

        return self

    def predict(self, features: np.ndarray) -> list:
        """Predict class labels using the SVM head."""
        features_pca = self._transform_preprocess(features)
        return list(self.svm.predict(features_pca))

    def predict_single(self, feature: np.ndarray) -> str:
        return self.predict(feature.reshape(1, -1))[0]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Per-class probability estimates from SVM (Platt scaling)."""
        features_pca = self._transform_preprocess(features)
        return self.svm.predict_proba(features_pca)

    def confidence(self, features: np.ndarray) -> list:
        """Returns max probability from SVM predict_proba."""
        return list(self.predict_proba(features).max(axis=1))

    def confidence_single(self, feature: np.ndarray) -> dict:
        """Returns {class: probability} dict for a single sample."""
        proba = self.predict_proba(feature.reshape(1, -1))[0]
        return dict(zip(self.svm.classes_, proba))

    def neuron_stats(self) -> dict:
        """Per-neuron label distribution — SOM topology view."""
        stats = {}
        for pos, lbls in self._raw_label_map.items():
            total = len(lbls)
            dist = {lbl: lbls.count(lbl) / total for lbl in set(lbls)}
            stats[pos] = {
                "dominant": self.winner_label_map[pos],
                "total": total,
                "distribution": dist,
            }
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path=None):
        if path is None:
            path = MODELS_DIR / "hybrid_cnn_som.pkl"
        joblib.dump({
            "scaler": self.scaler,
            "pca": self.pca,
            "som": self.som,
            "svm": self.svm,
            "winner_label_map": self.winner_label_map,
            "_raw_label_map": self._raw_label_map,
            "config": {
                "som_x": self.som_x,
                "som_y": self.som_y,
                "n_components": self.n_components,
                "sigma": self.sigma,
                "learning_rate": self.learning_rate,
                "num_iterations": self.num_iterations,
                "svm_C": self.svm_C,
                "svm_gamma": self.svm_gamma,
            },
        }, path)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path) -> "HybridCNNSOM":
        data = joblib.load(path)
        cfg = data["config"]
        model = cls(**cfg)
        model.scaler = data["scaler"]
        model.pca = data["pca"]
        model.som = data["som"]
        model.svm = data["svm"]
        model.winner_label_map = data["winner_label_map"]
        model._raw_label_map = data["_raw_label_map"]
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_preprocess(self, features: np.ndarray) -> np.ndarray:
        # float64 avoids overflow in PCA's randomized SVD matmul
        f64 = features.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scaled = self.scaler.fit_transform(f64)
            pca_out = self.pca.fit_transform(scaled)
        pca_out = np.nan_to_num(pca_out, nan=0.0, posinf=0.0, neginf=0.0)
        return pca_out.astype(np.float64)

    def _transform_preprocess(self, features: np.ndarray) -> np.ndarray:
        f64 = features.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scaled = self.scaler.transform(f64)
            pca_out = self.pca.transform(scaled)
        pca_out = np.nan_to_num(pca_out, nan=0.0, posinf=0.0, neginf=0.0)
        return pca_out.astype(np.float64)

    def _init_som_centroids(self, features_pca: np.ndarray, labels: np.ndarray):
        """
        Initialise SOM weights using per-class centroids.
        Neuron order follows sorted class order.
        Any classes beyond som_x*som_y neurons fall back to random init.
        """
        weights = self.som.get_weights().copy()  # (x, y, n_components)
        positions = [(x, y) for x in range(self.som_x) for y in range(self.som_y)]

        classes_in_data = sorted(set(labels))
        for idx, cls in enumerate(classes_in_data[:self.som_x * self.som_y]):
            mask = labels == cls
            centroid = features_pca[mask].mean(axis=0)
            x, y = positions[idx]
            weights[x, y] = centroid

        self.som._weights = weights
        print("  SOM initialised with class centroids: "
              + ", ".join(f"N{i+1}←{c}" for i, c in enumerate(classes_in_data[:self.som_x * self.som_y])))

    def _build_label_map(self, features_pca: np.ndarray, labels: np.ndarray):
        self._raw_label_map = {}
        for feat, label in zip(features_pca, labels):
            winner = self.som.winner(feat)
            self._raw_label_map.setdefault(winner, []).append(label)

        self.winner_label_map = {
            pos: max(set(lbls), key=lbls.count)
            for pos, lbls in self._raw_label_map.items()
        }
