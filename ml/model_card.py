from __future__ import annotations

def generate_model_card(
    *,
    training_timestamp: str,
    dataset_name: str,
    validation_metrics: dict,
    test_metrics: dict,
    validation_roc_auc: float,
    test_roc_auc: float,
    model_hash: str,
    environment_fingerprint: dict,
    selected_threshold: float,
) -> str:
    env_summary = environment_fingerprint.get("package_versions_used_by_training_script", {})
    return f"""# Model Card

## Overview
- Dataset name: {dataset_name}
- Target: `y` mapped as `yes -> 1`, `no -> 0`
- Split strategy: stratified 60/20/20 train/validation/test with `random_state=42`
- Model type: scikit-learn `Pipeline(ColumnTransformer + LogisticRegression)`
- Training timestamp: {training_timestamp}
- Artifact hash: `{model_hash}`

## Data Notes
- Leakage warning: `duration` is excluded because it is recorded after the call ends.
- `pdays == 999` handling: keep the original numeric `pdays` value and add `pdays_was_999`.
- Unknown handling: the string `unknown` is preserved as a legitimate category, not missing data.

## Preprocessing
- Numeric pipeline: `SimpleImputer(strategy="median")` then `StandardScaler()`
- Categorical pipeline: `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore")`

## Threshold Rule
- Threshold selection is performed on validation data only.
- Rule: choose the highest threshold where recall is at least `0.75`.
- Selected threshold: {selected_threshold:.2f}

## Validation Metrics
- ROC AUC: {validation_roc_auc:.4f}
- Accuracy: {validation_metrics["accuracy"]:.4f}
- Precision: {validation_metrics["precision"]:.4f}
- Recall: {validation_metrics["recall"]:.4f}
- F1: {validation_metrics["f1"]:.4f}

## Test Metrics
- ROC AUC: {test_roc_auc:.4f}
- Accuracy: {test_metrics["accuracy"]:.4f}
- Precision: {test_metrics["precision"]:.4f}
- Recall: {test_metrics["recall"]:.4f}
- F1: {test_metrics["f1"]:.4f}

## Limitations
- This phase is offline training only and does not include model registration or serving.
- Logistic regression offers interpretability and speed, but may miss non-linear relationships.
- Threshold tuning is optimized for a recall floor and may trade off precision.

## Ethical Caveat
- Marketing outreach predictions can influence who receives contact attempts. Review downstream use for fairness, disparate impact, and responsible customer treatment before deployment.

## Environment Fingerprint Summary
- Python: {environment_fingerprint.get("python_version", "").split()[0]}
- Operating system: {environment_fingerprint.get("operating_system", "unknown")}
- pandas: {env_summary.get("pandas", "unknown")}
- numpy: {env_summary.get("numpy", "unknown")}
- scikit-learn: {env_summary.get("scikit-learn", "unknown")}
- joblib: {env_summary.get("joblib", "unknown")}
"""
