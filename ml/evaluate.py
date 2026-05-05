from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_python(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [to_python(item) for item in value]
    if isinstance(value, tuple):
        return [to_python(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Series):
        return [to_python(item) for item in value.tolist()]
    return value


def compute_roc_auc(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def compute_accuracy(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_precision(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def compute_recall(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


def compute_f1(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> list[list[int]]:
    matrix = confusion_matrix(y_true, y_pred)
    return [[int(value) for value in row] for row in matrix.tolist()]


def compute_class_balance(y_true: pd.Series | np.ndarray) -> dict[str, float | int]:
    series = pd.Series(y_true)
    positives = int(series.sum())
    total = int(series.shape[0])
    negatives = int(total - positives)
    positive_rate = float(positives / total) if total else 0.0
    return {
        "total": total,
        "negative_count": negatives,
        "positive_count": positives,
        "positive_rate": positive_rate,
    }


def compute_positive_rate(y_pred: np.ndarray) -> float:
    predictions = np.asarray(y_pred)
    if predictions.size == 0:
        return 0.0
    return float(predictions.mean())


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    predictions = (np.asarray(y_prob) >= threshold).astype(int)
    return to_python(
        {
            "threshold": float(threshold),
            "accuracy": compute_accuracy(y_true, predictions),
            "precision": compute_precision(y_true, predictions),
            "recall": compute_recall(y_true, predictions),
            "f1": compute_f1(y_true, predictions),
            "positive_rate": compute_positive_rate(predictions),
            "confusion_matrix": compute_confusion_matrix(y_true, predictions),
        }
    )
