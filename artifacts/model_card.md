# Model Card

## Overview
- Dataset name: bank-additional-full.csv
- Target: `y` mapped as `yes -> 1`, `no -> 0`
- Split strategy: stratified 60/20/20 train/validation/test with `random_state=42`
- Model type: scikit-learn `Pipeline(ColumnTransformer + LogisticRegression)`
- Training timestamp: 2026-05-05T07:48:26.169937+00:00
- Artifact hash: `adad4a7ef9322233b13eeeccbde72c1b10c4f514bb78a1e113c0da7f0b799009`

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
- Selected threshold: 0.38

## Validation Metrics
- ROC AUC: 0.8017
- Accuracy: 0.7028
- Precision: 0.2399
- Recall: 0.7554
- F1: 0.3642

## Test Metrics
- ROC AUC: 0.8013
- Accuracy: 0.7034
- Precision: 0.2401
- Recall: 0.7543
- F1: 0.3643

## Limitations
- This phase is offline training only and does not include model registration or serving.
- Logistic regression offers interpretability and speed, but may miss non-linear relationships.
- Threshold tuning is optimized for a recall floor and may trade off precision.

## Ethical Caveat
- Marketing outreach predictions can influence who receives contact attempts. Review downstream use for fairness, disparate impact, and responsible customer treatment before deployment.

## Environment Fingerprint Summary
- Python: 3.12.3
- Operating system: Linux
- pandas: 2.3.3
- numpy: 2.4.4
- scikit-learn: 1.7.2
- joblib: 1.5.3
