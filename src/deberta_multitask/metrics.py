from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    cls_labels = labels[:, 0].astype(int)
    reg_labels = labels[:, 1].astype(float)

    cls_pred = logits.argmax(axis=-1)
    expected = (softmax(logits) * np.linspace(0, 1, logits.shape[-1])).sum(axis=-1)

    return {
        "accuracy": accuracy_score(cls_labels, cls_pred),
        "macro_f1": f1_score(cls_labels, cls_pred, average="macro"),
        "mae": mean_absolute_error(reg_labels, expected),
        "rmse": mean_squared_error(reg_labels, expected, squared=False),
    }


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)

