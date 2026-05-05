from __future__ import annotations

import numpy as np
import pandas as pd

from ml.evaluate import compute_accuracy, compute_f1, compute_precision, compute_recall


def select_highest_threshold_for_recall(
    y_true,
    y_prob,
    min_recall: float = 0.75,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    """Select the highest threshold whose recall satisfies the minimum rule."""
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows: list[dict[str, float]] = []

    probabilities = np.asarray(y_prob)
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": compute_precision(y_true, predictions),
                "recall": compute_recall(y_true, predictions),
                "f1": compute_f1(y_true, predictions),
                "accuracy": compute_accuracy(y_true, predictions),
            }
        )

    threshold_table = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "accuracy"])
    valid_rows = threshold_table[threshold_table["recall"] >= min_recall]
    if valid_rows.empty:
        raise ValueError(
            f"No threshold in the range 0.01 to 0.99 satisfied recall >= {min_recall:.2f}."
        )

    selected_row = valid_rows.sort_values("threshold", ascending=False).iloc[0]
    selected_threshold = float(selected_row["threshold"])
    selected_metrics = {
        "threshold": float(selected_row["threshold"]),
        "precision": float(selected_row["precision"]),
        "recall": float(selected_row["recall"]),
        "f1": float(selected_row["f1"]),
        "accuracy": float(selected_row["accuracy"]),
    }
    return selected_threshold, selected_metrics, threshold_table

