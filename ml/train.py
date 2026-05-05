from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.data import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    DROPPED_COLUMNS,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_MAPPING,
    build_preprocessor,
    load_raw_data,
    prepare_features_and_target,
    split_data,
)
from ml.evaluate import compute_class_balance, compute_roc_auc, metrics_at_threshold, to_python
from ml.fingerprint import save_environment_fingerprint, save_model_hash
from ml.model_card import generate_model_card
from ml.schemas import create_schema_dict, save_schema
from ml.threshold import select_highest_threshold_for_recall

MODEL_FILENAME = "model.joblib"
THRESHOLD_JSON_FILENAME = "threshold.json"
THRESHOLD_TABLE_FILENAME = "threshold_table.csv"
METRICS_FILENAME = "metrics.json"
SCHEMA_FILENAME = "schema.json"
MODEL_CARD_FILENAME = "model_card.md"
ENVIRONMENT_FINGERPRINT_FILENAME = "environment_fingerprint.json"
MODEL_HASH_FILENAME = "model_hash.txt"
MIN_RECALL_RULE = 0.75


def build_training_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_python(payload), indent=2), encoding="utf-8")


def run_training(
    dataset_path: Path = DATASET_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict:
    dataset_path = Path(dataset_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    training_timestamp = datetime.now(timezone.utc).isoformat()

    raw_data = load_raw_data(dataset_path)
    X, y = prepare_features_and_target(raw_data)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train)

    validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
    selected_threshold, selected_threshold_metrics, threshold_table = (
        select_highest_threshold_for_recall(
            y_true=y_validation,
            y_prob=validation_probabilities,
            min_recall=MIN_RECALL_RULE,
        )
    )

    validation_metrics = metrics_at_threshold(
        y_true=y_validation,
        y_prob=validation_probabilities,
        threshold=selected_threshold,
    )
    validation_roc_auc = compute_roc_auc(y_validation, validation_probabilities)

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(
        y_true=y_test,
        y_prob=test_probabilities,
        threshold=selected_threshold,
    )
    test_roc_auc = compute_roc_auc(y_test, test_probabilities)

    model_path = artifacts_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)

    threshold_table_path = artifacts_dir / THRESHOLD_TABLE_FILENAME
    threshold_table.to_csv(threshold_table_path, index=False)

    selected_threshold_row = (
        threshold_table[threshold_table["threshold"].round(2) == round(selected_threshold, 2)]
        .iloc[0]
        .to_dict()
    )
    threshold_payload = {
        "selected_threshold": float(selected_threshold),
        "min_recall_rule": float(MIN_RECALL_RULE),
        "selected_threshold_metrics": selected_threshold_metrics,
        "selected_threshold_row": {
            key: float(value) for key, value in selected_threshold_row.items()
        },
        "validation_metrics_at_threshold": validation_metrics,
        "created_at": training_timestamp,
    }
    save_json(artifacts_dir / THRESHOLD_JSON_FILENAME, threshold_payload)

    schema = create_schema_dict()
    save_schema(schema, artifacts_dir / SCHEMA_FILENAME)

    environment_fingerprint = save_environment_fingerprint(
        artifacts_dir / ENVIRONMENT_FINGERPRINT_FILENAME
    )
    model_hash = save_model_hash(model_path, artifacts_dir / MODEL_HASH_FILENAME)

    metrics_payload = {
        "selected_threshold": float(selected_threshold),
        "validation_metrics_at_selected_threshold": validation_metrics,
        "test_metrics_at_selected_threshold": test_metrics,
        "validation_roc_auc": float(validation_roc_auc),
        "test_roc_auc": float(test_roc_auc),
        "class_balance": {
            "train": compute_class_balance(y_train),
            "validation": compute_class_balance(y_validation),
            "test": compute_class_balance(y_test),
        },
        "model_parameters": pipeline.named_steps["classifier"].get_params(),
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
        "dataset_path": str(dataset_path.resolve()),
        "dropped_columns": list(DROPPED_COLUMNS),
        "target_mapping": dict(TARGET_MAPPING),
        "training_timestamp": training_timestamp,
        "model_hash": model_hash,
    }
    save_json(artifacts_dir / METRICS_FILENAME, metrics_payload)

    model_card_text = generate_model_card(
        training_timestamp=training_timestamp,
        dataset_name=dataset_path.name,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_roc_auc=validation_roc_auc,
        test_roc_auc=test_roc_auc,
        model_hash=model_hash,
        environment_fingerprint=environment_fingerprint,
        selected_threshold=selected_threshold,
    )
    (artifacts_dir / MODEL_CARD_FILENAME).write_text(model_card_text, encoding="utf-8")

    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_path": str(dataset_path.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "selected_threshold": float(selected_threshold),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_roc_auc": float(validation_roc_auc),
        "test_roc_auc": float(test_roc_auc),
        "model_hash": model_hash,
    }


def main() -> None:
    results = run_training()
    print(f"selected threshold: {results['selected_threshold']:.2f}")
    print(f"validation recall: {results['validation_metrics']['recall']:.4f}")
    print(f"validation F1: {results['validation_metrics']['f1']:.4f}")
    print(f"test recall: {results['test_metrics']['recall']:.4f}")
    print(f"test F1: {results['test_metrics']['f1']:.4f}")
    print(f"test ROC AUC: {results['test_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
