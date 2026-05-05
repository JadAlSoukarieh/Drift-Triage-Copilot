from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.data import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    MODEL_INPUT_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    load_raw_data,
    prepare_features_and_target,
)
from ml.evaluate import to_python
from ml.train import (
    ENVIRONMENT_FINGERPRINT_FILENAME,
    METRICS_FILENAME,
    MODEL_CARD_FILENAME,
    MODEL_FILENAME,
    MODEL_HASH_FILENAME,
    SCHEMA_FILENAME,
    THRESHOLD_JSON_FILENAME,
    THRESHOLD_TABLE_FILENAME,
)

DEFAULT_TRACKING_URI = "file:./mlruns"
MLFLOW_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
EXPERIMENT_NAME = "drift-triage-bank-marketing"
REGISTERED_MODEL_NAME = "bank-marketing-classifier"

REQUIRED_ARTIFACT_FILENAMES = [
    MODEL_FILENAME,
    THRESHOLD_JSON_FILENAME,
    METRICS_FILENAME,
    SCHEMA_FILENAME,
    MODEL_CARD_FILENAME,
    ENVIRONMENT_FINGERPRINT_FILENAME,
    MODEL_HASH_FILENAME,
    THRESHOLD_TABLE_FILENAME,
]

ARTIFACT_LOG_FILENAMES = [
    SCHEMA_FILENAME,
    MODEL_CARD_FILENAME,
    THRESHOLD_JSON_FILENAME,
    THRESHOLD_TABLE_FILENAME,
    METRICS_FILENAME,
    ENVIRONMENT_FINGERPRINT_FILENAME,
    MODEL_HASH_FILENAME,
]

def get_tracking_uri() -> str:
    return os.getenv(MLFLOW_TRACKING_URI_ENV, DEFAULT_TRACKING_URI)


def load_json(path: Path) -> dict[str, Any]:
    return to_python(json.loads(path.read_text(encoding="utf-8")))


def require_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, Path]:
    artifacts_dir = Path(artifacts_dir)
    artifact_paths = {
        filename: artifacts_dir / filename for filename in REQUIRED_ARTIFACT_FILENAMES
    }
    missing_paths = [
        str(path.resolve()) for path in artifact_paths.values() if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required training artifacts for MLflow registration: "
            + ", ".join(missing_paths)
            + ". Run `python -m ml.train` first."
        )
    return artifact_paths


def extract_logged_params(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    model_parameters = metrics_payload["model_parameters"]
    split_sizes = metrics_payload["split_sizes"]
    params = {
        "C": model_parameters["C"],
        "class_weight": model_parameters["class_weight"],
        "max_iter": model_parameters["max_iter"],
        "random_state": model_parameters["random_state"],
        "solver": model_parameters["solver"],
        "penalty": model_parameters["penalty"],
        "selected_threshold": metrics_payload["selected_threshold"],
        "train_size": split_sizes["train"],
        "validation_size": split_sizes["validation"],
        "test_size": split_sizes["test"],
    }
    return to_python(params)


def extract_logged_metrics(metrics_payload: dict[str, Any]) -> dict[str, float]:
    validation_metrics = metrics_payload["validation_metrics_at_selected_threshold"]
    test_metrics = metrics_payload["test_metrics_at_selected_threshold"]
    metrics = {
        "validation_roc_auc": metrics_payload["validation_roc_auc"],
        "test_roc_auc": metrics_payload["test_roc_auc"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1"],
        "validation_positive_rate": validation_metrics["positive_rate"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_positive_rate": test_metrics["positive_rate"],
    }
    return {key: float(value) for key, value in to_python(metrics).items()}


def build_run_tags(model_hash: str) -> dict[str, str]:
    return {
        "project": "drift-triage-copilot",
        "dataset": "uci-bank-marketing",
        "model_family": "logistic_regression",
        "target": "term_deposit_subscription",
        "threshold_rule": "highest_threshold_with_validation_recall_gte_0.75",
        "leakage_column_dropped": "duration",
        "pdays_sentinel": "pdays_was_999",
        "unknown_handling": "preserved_as_category",
        "model_hash": model_hash,
    }


def build_model_version_tags(
    metrics_payload: dict[str, Any],
    model_hash: str,
) -> dict[str, str]:
    return {
        "validation_recall": str(
            metrics_payload["validation_metrics_at_selected_threshold"]["recall"]
        ),
        "test_recall": str(metrics_payload["test_metrics_at_selected_threshold"]["recall"]),
        "test_f1": str(metrics_payload["test_metrics_at_selected_threshold"]["f1"]),
        "test_roc_auc": str(metrics_payload["test_roc_auc"]),
        "selected_threshold": str(metrics_payload["selected_threshold"]),
        "model_hash": model_hash,
        "status": "candidate",
    }


def log_params(mlflow_module: Any, params: dict[str, Any]) -> None:
    for key, value in params.items():
        mlflow_module.log_param(key, value)


def log_metrics(mlflow_module: Any, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        mlflow_module.log_metric(key, float(value))


def log_artifacts(mlflow_module: Any, artifact_paths: dict[str, Path]) -> None:
    for filename in ARTIFACT_LOG_FILENAMES:
        mlflow_module.log_artifact(str(artifact_paths[filename]))


def get_mlflow_modules() -> tuple[Any, Any, Any]:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    from mlflow.models import infer_signature

    return mlflow, MlflowClient, MlflowException, infer_signature


def build_input_example() -> pd.DataFrame | None:
    try:
        raw_data = load_raw_data(DATASET_PATH)
        features, _ = prepare_features_and_target(raw_data.head(1))
    except (FileNotFoundError, ValueError, KeyError):
        return None

    input_example = features.head(1).copy()[MODEL_INPUT_FEATURES]
    input_example[NUMERIC_FEATURES] = input_example[NUMERIC_FEATURES].astype(float)
    input_example[CATEGORICAL_FEATURES] = input_example[CATEGORICAL_FEATURES].astype(str)
    return input_example


def ensure_registered_model(client: Any, registered_model_name: str, mlflow_exception: type[Exception]) -> None:
    try:
        client.get_registered_model(registered_model_name)
    except mlflow_exception:
        client.create_registered_model(registered_model_name)


def register_model(
    mlflow_module: Any,
    client: Any,
    model_path: Path,
    registered_model_name: str,
    model_version_tags: dict[str, str],
    infer_signature_fn: Any,
) -> tuple[str, str]:
    model = joblib.load(model_path)
    input_example = build_input_example()
    log_model_kwargs = {
        "sk_model": model,
        "artifact_path": "model",
    }
    if input_example is not None:
        log_model_kwargs["input_example"] = input_example
        log_model_kwargs["signature"] = infer_signature_fn(
            input_example,
            model.predict(input_example),
        )

    model_info = mlflow_module.sklearn.log_model(**log_model_kwargs)
    registered_model = mlflow_module.register_model(
        model_uri=model_info.model_uri,
        name=registered_model_name,
        await_registration_for=60,
    )
    model_version = str(registered_model.version)

    for key, value in model_version_tags.items():
        client.set_model_version_tag(registered_model_name, model_version, key, value)

    if hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(
            registered_model_name,
            "candidate",
            model_version,
        )

    return model_info.model_uri, model_version


def run_registration(
    artifacts_dir: Path = ARTIFACTS_DIR,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    artifact_paths = require_artifacts(artifacts_dir)
    metrics_payload = load_json(artifact_paths[METRICS_FILENAME])
    model_hash = artifact_paths[MODEL_HASH_FILENAME].read_text(encoding="utf-8").strip()

    tracking_uri = tracking_uri or get_tracking_uri()
    mlflow, mlflow_client_cls, mlflow_exception, infer_signature_fn = get_mlflow_modules()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow_client_cls(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    ensure_registered_model(client, REGISTERED_MODEL_NAME, mlflow_exception)

    params = extract_logged_params(metrics_payload)
    metrics = extract_logged_metrics(metrics_payload)
    tags = build_run_tags(model_hash)
    model_version_tags = build_model_version_tags(metrics_payload, model_hash)

    with mlflow.start_run(run_name="bank-marketing-registration") as run:
        mlflow.set_tags(tags)
        log_params(mlflow, params)
        log_metrics(mlflow, metrics)
        log_artifacts(mlflow, artifact_paths)
        _, model_version = register_model(
            mlflow_module=mlflow,
            client=client,
            model_path=artifact_paths[MODEL_FILENAME],
            registered_model_name=REGISTERED_MODEL_NAME,
            model_version_tags=model_version_tags,
            infer_signature_fn=infer_signature_fn,
        )

    return {
        "project_root": str(PROJECT_ROOT),
        "tracking_uri": tracking_uri,
        "run_id": run.info.run_id,
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "selected_threshold": float(metrics_payload["selected_threshold"]),
        "test_roc_auc": float(metrics_payload["test_roc_auc"]),
        "test_f1": float(metrics_payload["test_metrics_at_selected_threshold"]["f1"]),
        "model_hash": model_hash,
        "status": "candidate",
    }


def main() -> None:
    results = run_registration()
    print(f"run_id: {results['run_id']}")
    print(f"registered model name: {results['registered_model_name']}")
    print(f"model version: {results['model_version']}")
    print(f"selected threshold: {results['selected_threshold']:.2f}")
    print(f"test ROC AUC: {results['test_roc_auc']:.4f}")
    print(f"test F1: {results['test_f1']:.4f}")
    print(f"model hash: {results['model_hash']}")
    print("status: candidate")
    print("production alias/stage is intentionally not set in this phase.")


if __name__ == "__main__":
    main()
