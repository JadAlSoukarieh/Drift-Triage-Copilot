from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from services.model_service.app.core.settings import get_settings
from services.model_service.app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from services.model_service.app.services.model_loader import get_model, get_threshold


def predict_subscription(request: PredictionRequest) -> PredictionResponse:
    settings = get_settings()
    request_id = str(uuid4())
    features = request.to_model_dataframe()
    model = get_model()
    threshold = get_threshold()

    probability = float(model.predict_proba(features)[:, 1][0])
    prediction = int(probability >= threshold)
    label = "yes" if prediction == 1 else "no"
    created_at = datetime.now(timezone.utc).isoformat()

    return PredictionResponse(
        request_id=request_id,
        model_name=settings.model_name,
        model_alias=settings.model_alias,
        prediction=prediction,
        label=label,
        probability=probability,
        threshold=float(threshold),
        created_at=created_at,
    )

