from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "bank-additional-full.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_COLUMN = "y"
INTERNAL_TARGET_COLUMN = "y_binary"
DROPPED_COLUMNS = ["duration", TARGET_COLUMN, INTERNAL_TARGET_COLUMN]
TARGET_MAPPING = {"yes": 1, "no": 0}

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "pdays_was_999",
]

MODEL_INPUT_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "pdays_was_999",
]

RAW_INPUT_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

REQUIRED_SOURCE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    TARGET_COLUMN,
]

RANDOM_STATE = 42


def load_raw_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the bank marketing dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Bank marketing dataset not found at {path.resolve()}. "
            "Expected data/bank-additional-full.csv."
        )

    dataframe = pd.read_csv(path, sep=";")
    validate_required_columns(dataframe)
    return dataframe


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )


def prepare_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare training features and binary target without leakage."""
    prepared = dataframe.copy()
    prepared[INTERNAL_TARGET_COLUMN] = prepared[TARGET_COLUMN].map(TARGET_MAPPING)
    prepared["pdays_was_999"] = (prepared["pdays"] == 999).astype(int)

    if prepared[INTERNAL_TARGET_COLUMN].isna().any():
        unknown_targets = sorted(
            prepared.loc[prepared[INTERNAL_TARGET_COLUMN].isna(), TARGET_COLUMN]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Dataset contains unmapped target values: " + ", ".join(unknown_targets)
        )

    y = prepared[INTERNAL_TARGET_COLUMN].astype(int)
    X = prepared.drop(columns=[column for column in DROPPED_COLUMNS if column in prepared])
    X = X[MODEL_INPUT_FEATURES]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified 60/20/20 train/validation/test splits."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=random_state,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
