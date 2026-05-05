from __future__ import annotations

import json
from pathlib import Path

from ml.data import (
    CATEGORICAL_FEATURES,
    DROPPED_COLUMNS,
    MODEL_INPUT_FEATURES,
    NUMERIC_FEATURES,
    RAW_INPUT_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING,
)


def create_schema_dict() -> dict:
    return {
        "expected_input_fields": list(RAW_INPUT_FEATURES),
        "required_features_in_order": list(MODEL_INPUT_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "target_mapping": dict(TARGET_MAPPING),
        "dropped_columns": list(DROPPED_COLUMNS),
        "pdays_handling": {
            "keep_original_numeric_column": True,
            "sentinel_value": 999,
            "derived_feature": "pdays_was_999",
            "replace_with_nan": False,
        },
        "unknown_category_handling": {
            "treat_unknown_as_missing": False,
            "value": "unknown",
            "notes": "The literal string 'unknown' is preserved as a valid category.",
        },
        "prediction_output_schema": {
            "predicted_label": {
                "type": "int",
                "description": "Binary prediction after applying the selected threshold.",
            },
            "positive_class_probability": {
                "type": "float",
                "description": "Predicted probability that the target y is 'yes'.",
            },
        },
        "probability_output_meaning": "The probability output represents P(y='yes' | input features).",
        "model_input_notes": [
            "Do not provide duration; it is excluded because it leaks post-call information.",
            "Provide pdays as the original numeric field, including the sentinel value 999.",
            "Provide the literal category 'unknown' unchanged where it appears.",
            "The training pipeline derives pdays_was_999 internally from the raw pdays field.",
        ],
    }


def save_schema(schema: dict, path: Path) -> None:
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
