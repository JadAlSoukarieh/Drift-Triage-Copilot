from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from services.model_service.app.schemas.prediction import PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "schema.json"


def valid_payload(**overrides):
    payload = {
        "age": 42,
        "job": "technician",
        "marital": "married",
        "education": "professional.course",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": 1.1,
        "cons_price_idx": 93.994,
        "cons_conf_idx": -36.4,
        "euribor3m": 4.857,
        "nr_employed": 5191.0,
    }
    payload.update(overrides)
    return payload


class PredictionValidationTests(unittest.TestCase):
    def test_valid_request_converts_to_expected_dataframe(self) -> None:
        request = PredictionRequest(**valid_payload())
        dataframe = request.to_model_dataframe()

        self.assertEqual(dataframe.loc[0, "emp.var.rate"], 1.1)
        self.assertEqual(dataframe.loc[0, "cons.price.idx"], 93.994)
        self.assertEqual(dataframe.loc[0, "cons.conf.idx"], -36.4)
        self.assertEqual(dataframe.loc[0, "nr.employed"], 5191.0)
        self.assertEqual(dataframe.loc[0, "pdays_was_999"], 1)
        self.assertNotIn("duration", dataframe.columns)
        self.assertNotIn("y", dataframe.columns)
        self.assertNotIn("y_binary", dataframe.columns)

    def test_pdays_not_999_creates_zero_indicator(self) -> None:
        request = PredictionRequest(**valid_payload(pdays=10))
        dataframe = request.to_model_dataframe()
        self.assertEqual(dataframe.loc[0, "pdays_was_999"], 0)

    def test_unknown_is_accepted(self) -> None:
        request = PredictionRequest(**valid_payload(job="unknown"))
        self.assertEqual(request.job, "unknown")

    def test_missing_required_field_fails_validation(self) -> None:
        with self.assertRaises(ValidationError):
            PredictionRequest(**valid_payload(age=None))

    def test_forbidden_fields_are_rejected(self) -> None:
        for field_name, value in (
            ("duration", 100),
            ("y", "yes"),
            ("y_binary", 1),
            ("pdays_was_999", 1),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValidationError):
                    PredictionRequest(**valid_payload(**{field_name: value}))

    def test_feature_order_matches_schema_artifact(self) -> None:
        request = PredictionRequest(**valid_payload())
        dataframe = request.to_model_dataframe()
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            list(dataframe.columns),
            schema["required_features_in_order"],
        )


if __name__ == "__main__":
    unittest.main()
