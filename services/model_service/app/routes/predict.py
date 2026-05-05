from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.model_service.app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from services.model_service.app.services.prediction_service import predict_subscription

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return predict_subscription(request)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_unavailable",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "message": "Prediction failed.",
            },
        ) from exc

