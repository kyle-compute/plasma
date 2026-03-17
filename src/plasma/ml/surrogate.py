"""Torch-optional scalar surrogate for run-summary regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from plasma.io.reports import save_json_model


class ScalarSurrogateConfig(BaseModel):
    """Hyperparameters for the scalar surrogate."""

    hidden_layers: list[int] = Field(default_factory=lambda: [64, 64])
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 32


@dataclass
class _Normalization:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


class SavedScalarSurrogateMetadata(BaseModel):
    """Serializable metadata required to reload a trained surrogate."""

    feature_names: list[str]
    target_names: list[str]
    config: ScalarSurrogateConfig
    x_mean: list[float]
    x_std: list[float]
    y_mean: list[float]
    y_std: list[float]


class ScalarSurrogate:
    """Simple MLP surrogate over scalar run-summary datasets."""

    def __init__(self, feature_names: list[str], target_names: list[str], config: ScalarSurrogateConfig | None = None):
        self.feature_names = feature_names
        self.target_names = target_names
        self.config = config or ScalarSurrogateConfig()
        self._model = None
        self._normalization: _Normalization | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        torch, nn = _require_torch()
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Features and targets must both be rank-2 arrays.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Features and targets must have the same number of rows.")
        if x.shape[1] != len(self.feature_names):
            raise ValueError("Feature width does not match feature_names.")
        if y.shape[1] != len(self.target_names):
            raise ValueError("Target width does not match target_names.")

        x_mean = x.mean(axis=0)
        x_std = np.where(x.std(axis=0) > 0.0, x.std(axis=0), 1.0)
        y_mean = y.mean(axis=0)
        y_std = np.where(y.std(axis=0) > 0.0, y.std(axis=0), 1.0)
        self._normalization = _Normalization(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

        x_norm = torch.as_tensor((x - x_mean) / x_std)
        y_norm = torch.as_tensor((y - y_mean) / y_std)

        model = _build_model(nn, x.shape[1], y.shape[1], self.config)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            prediction = model(x_norm)
            loss = loss_fn(prediction, y_norm)
            loss.backward()
            optimizer.step()

        self._model = model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        torch, _nn = _require_torch()
        if self._model is None or self._normalization is None:
            raise RuntimeError("Surrogate has not been fit yet.")
        x = np.asarray(features, dtype=np.float32)
        x_norm = (x - self._normalization.x_mean) / self._normalization.x_std
        with torch.no_grad():
            y_norm = self._model(torch.as_tensor(x_norm)).cpu().numpy()
        return (y_norm * self._normalization.y_std) + self._normalization.y_mean

    def save(self, directory: str | Path) -> None:
        torch, _nn = _require_torch()
        if self._model is None or self._normalization is None:
            raise RuntimeError("Surrogate has not been fit yet.")
        target_dir = Path(directory).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        save_json_model(
            target_dir / "surrogate_metadata.json",
            SavedScalarSurrogateMetadata(
                feature_names=self.feature_names,
                target_names=self.target_names,
                config=self.config,
                x_mean=self._normalization.x_mean.tolist(),
                x_std=self._normalization.x_std.tolist(),
                y_mean=self._normalization.y_mean.tolist(),
                y_std=self._normalization.y_std.tolist(),
            ),
        )
        torch.save(self._model.state_dict(), target_dir / "surrogate_model.pt")

    @classmethod
    def load(cls, directory: str | Path) -> ScalarSurrogate:
        torch, nn = _require_torch()
        target_dir = Path(directory).resolve()
        metadata = SavedScalarSurrogateMetadata.model_validate_json(
            (target_dir / "surrogate_metadata.json").read_text()
        )
        surrogate = cls(
            feature_names=metadata.feature_names,
            target_names=metadata.target_names,
            config=metadata.config,
        )
        surrogate._normalization = _Normalization(
            x_mean=np.asarray(metadata.x_mean, dtype=np.float32),
            x_std=np.asarray(metadata.x_std, dtype=np.float32),
            y_mean=np.asarray(metadata.y_mean, dtype=np.float32),
            y_std=np.asarray(metadata.y_std, dtype=np.float32),
        )
        surrogate._model = _build_model(
            nn,
            len(metadata.feature_names),
            len(metadata.target_names),
            metadata.config,
        )
        surrogate._model.load_state_dict(torch.load(target_dir / "surrogate_model.pt", map_location="cpu"))
        surrogate._model = surrogate._model.eval()
        return surrogate


def _build_model(nn, n_features: int, n_targets: int, config: ScalarSurrogateConfig):
    layers: list[nn.Module] = []
    in_features = n_features
    for hidden in config.hidden_layers:
        layers.append(nn.Linear(in_features, hidden))
        layers.append(nn.GELU())
        in_features = hidden
    layers.append(nn.Linear(in_features, n_targets))
    return nn.Sequential(*layers)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is not installed. Install the optional ML dependencies with `pip install -e .[ml]`."
        ) from exc
    return torch, nn
