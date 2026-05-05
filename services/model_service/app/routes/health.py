from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.model_service.app.core.settings import get_settings
from services.model_service.app.services.model_loader import get_model, get_threshold

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "model_service",
    }


@router.get("/health/model")
def health_model() -> dict[str, str | float]:
    settings = get_settings()
    try:
        get_model()
        threshold = get_threshold()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_unavailable",
                "message": str(exc),
            },
        ) from exc

    return {
        "status": "ok",
        "service": "model_service",
        "model_name": settings.model_name,
        "model_alias": settings.model_alias,
        "threshold": float(threshold),
    }

