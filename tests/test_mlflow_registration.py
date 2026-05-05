from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.data import CATEGORICAL_FEATURES, MODEL_INPUT_FEATURES, NUMERIC_FEATURES
from ml.register_mlflow import (
    build_input_example,
    extract_logged_metrics,
    require_artifacts,
)


class RequireArtifactsTests(unittest.TestCase):
    def test_require_artifacts_fails_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts_dir = Path(temporary_directory)

            with self.assertRaises(FileNotFoundError) as context:
                require_artifacts(artifacts_dir)

            error_message = str(context.exception)
            self.assertIn("Run `python -m ml.train` first.", error_message)
            self.assertIn("model.joblib", error_message)


class MetricExtractionTests(unittest.TestCase):
    def test_extract_logged_metrics_returns_expected_keys(self) -> None:
        sample_metrics = {
            "selected_threshold": 0.38,
            "validation_roc_auc": 0.81,
            "test_roc_auc": 0.80,
            "validation_metrics_at_selected_threshold": {
                "accuracy": 0.70,
                "precision": 0.24,
                "recall": 0.76,
                "f1": 0.36,
                "positive_rate": 0.35,
            },
            "test_metrics_at_selected_threshold": {
                "accuracy": 0.71,
                "precision": 0.25,
                "recall": 0.75,
                "f1": 0.37,
                "positive_rate": 0.34,
            },
        }

        extracted_metrics = extract_logged_metrics(sample_metrics)

        expected_keys = {
            "validation_roc_auc",
            "test_roc_auc",
            "validation_accuracy",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_positive_rate",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_positive_rate",
        }

        self.assertEqual(set(extracted_metrics.keys()), expected_keys)
        self.assertEqual(extracted_metrics["test_f1"], 0.37)


class InputExampleTests(unittest.TestCase):
    def test_build_input_example_uses_expected_columns_and_dtypes(self) -> None:
        input_example = build_input_example()

        self.assertIsNotNone(input_example)
        self.assertEqual(list(input_example.columns), MODEL_INPUT_FEATURES)

        for column in NUMERIC_FEATURES:
            self.assertEqual(input_example[column].dtype.kind, "f", column)

        for column in CATEGORICAL_FEATURES:
            self.assertIn(input_example[column].dtype.kind, {"O", "U"}, column)


if __name__ == "__main__":
    unittest.main()
