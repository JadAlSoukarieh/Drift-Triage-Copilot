from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.data import DATASET_PATH, load_raw_data, prepare_features_and_target
from ml.threshold import select_highest_threshold_for_recall
from ml.train import run_training


class DataPreparationTests(unittest.TestCase):
    def test_prepare_data_excludes_leakage_columns(self) -> None:
        raw_data = load_raw_data(DATASET_PATH)
        X, _ = prepare_features_and_target(raw_data)

        self.assertNotIn("duration", X.columns)
        self.assertNotIn("y", X.columns)
        self.assertNotIn("y_binary", X.columns)

    def test_prepare_data_adds_pdays_was_999(self) -> None:
        raw_data = load_raw_data(DATASET_PATH)
        X, _ = prepare_features_and_target(raw_data)

        self.assertIn("pdays_was_999", X.columns)


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_highest_threshold_that_meets_recall(self) -> None:
        y_true = [1, 1, 0, 0]
        y_prob = [0.90, 0.80, 0.70, 0.10]

        threshold, metrics, _ = select_highest_threshold_for_recall(
            y_true=y_true,
            y_prob=y_prob,
            min_recall=0.75,
        )

        self.assertAlmostEqual(threshold, 0.80)
        self.assertAlmostEqual(metrics["recall"], 1.0)


class TrainingArtifactTests(unittest.TestCase):
    def test_run_training_creates_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts_dir = Path(temporary_directory)
            run_training(dataset_path=DATASET_PATH, artifacts_dir=artifacts_dir)

            expected_files = [
                "model.joblib",
                "threshold.json",
                "threshold_table.csv",
                "metrics.json",
                "schema.json",
                "model_card.md",
                "environment_fingerprint.json",
                "model_hash.txt",
            ]

            for filename in expected_files:
                self.assertTrue((artifacts_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
