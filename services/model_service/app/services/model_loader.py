from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import mlflow

from services.model_service.app.core.settings import get_settings


@lru_cache(maxsize=1)
def get_model() -> Any:
    settings = get_settings()
    model_uri = f"models:/{settings.model_name}@{settings.model_alias}"
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)

    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load MLflow model "
            f"name={settings.model_name} alias={settings.model_alias} "
            f"tracking_uri={settings.mlflow_tracking_uri}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def get_threshold() -> float:
    settings = get_settings()
    threshold_path = settings.threshold_path
    if not threshold_path.exists():
        raise RuntimeError("Threshold artifact missing. Run python -m ml.train first.")

    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    if "selected_threshold" not in payload:
        raise RuntimeError(
            f"Threshold artifact at {threshold_path} is missing selected_threshold."
        )
    return float(payload["selected_threshold"])

