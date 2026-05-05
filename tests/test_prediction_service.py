from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from services.model_service.app.schemas.prediction import PredictionRequest
from services.model_service.app.services.prediction_service import predict_subscription


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


class FakeModel:
    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, _):
        return self._probabilities


class PredictionServiceTests(unittest.TestCase):
    @patch("services.model_service.app.services.prediction_service.get_settings")
    @patch("services.model_service.app.services.prediction_service.get_threshold")
    @patch("services.model_service.app.services.prediction_service.get_model")
    def test_positive_prediction_response(
        self,
        mock_get_model,
        mock_get_threshold,
        mock_get_settings,
    ) -> None:
        mock_get_model.return_value = FakeModel([[0.2, 0.8]])
        mock_get_threshold.return_value = 0.38
        mock_get_settings.return_value.model_name = "bank-marketing-classifier"
        mock_get_settings.return_value.model_alias = "candidate"

        response = predict_subscription(PredictionRequest(**valid_payload()))

        self.assertEqual(response.prediction, 1)
        self.assertEqual(response.label, "yes")
        self.assertEqual(response.probability, 0.8)
        self.assertEqual(response.threshold, 0.38)
        self.assertEqual(response.model_name, "bank-marketing-classifier")
        self.assertEqual(response.model_alias, "candidate")
        self.assertTrue(response.request_id)
        self.assertTrue(response.created_at)

    @patch("services.model_service.app.services.prediction_service.get_settings")
    @patch("services.model_service.app.services.prediction_service.get_threshold")
    @patch("services.model_service.app.services.prediction_service.get_model")
    def test_negative_prediction_response(
        self,
        mock_get_model,
        mock_get_threshold,
        mock_get_settings,
    ) -> None:
        mock_get_model.return_value = FakeModel([[0.8, 0.2]])
        mock_get_threshold.return_value = 0.38
        mock_get_settings.return_value.model_name = "bank-marketing-classifier"
        mock_get_settings.return_value.model_alias = "candidate"

        response = predict_subscription(PredictionRequest(**valid_payload()))

        self.assertEqual(response.prediction, 0)
        self.assertEqual(response.label, "no")
        self.assertEqual(response.probability, 0.2)
        self.assertEqual(response.threshold, 0.38)
        self.assertTrue(response.request_id)
        self.assertTrue(response.created_at)


if __name__ == "__main__":
    unittest.main()
