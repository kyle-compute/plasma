"""Tests for the scalar surrogate wrapper."""

from __future__ import annotations

import numpy as np

from plasma.ml.surrogate import ScalarSurrogate, ScalarSurrogateConfig


def test_scalar_surrogate_fit_predict_and_reload(tmp_path) -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    targets = np.asarray([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    surrogate = ScalarSurrogate(
        feature_names=["x0"],
        target_names=["y0"],
        config=ScalarSurrogateConfig(hidden_layers=[8], epochs=50, learning_rate=5e-2),
    )

    surrogate.fit(features, targets)
    prediction = surrogate.predict(np.asarray([[4.0]], dtype=np.float32))
    assert prediction.shape == (1, 1)
    assert prediction[0, 0] > 6.0

    model_dir = tmp_path / "surrogate"
    surrogate.save(model_dir)
    loaded = ScalarSurrogate.load(model_dir)
    loaded_prediction = loaded.predict(np.asarray([[4.0]], dtype=np.float32))
    assert loaded_prediction.shape == (1, 1)
    assert abs(loaded_prediction[0, 0] - prediction[0, 0]) < 1e-3
