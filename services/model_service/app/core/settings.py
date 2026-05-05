from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    mlflow_tracking_uri: str = Field(default="file:./mlruns", alias="MLFLOW_TRACKING_URI")
    model_name: str = Field(default="bank-marketing-classifier", alias="MODEL_NAME")
    model_alias: str = Field(default="candidate", alias="MODEL_ALIAS")
    threshold_artifact_path: str = Field(
        default="artifacts/threshold.json",
        alias="THRESHOLD_ARTIFACT_PATH",
    )
    schema_artifact_path: str = Field(
        default="artifacts/schema.json",
        alias="SCHEMA_ARTIFACT_PATH",
    )
    env: str = Field(default="local", alias="ENV")

    @staticmethod
    def resolve_repo_path(path_value: str) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        return PROJECT_ROOT / candidate

    @property
    def threshold_path(self) -> Path:
        return self.resolve_repo_path(self.threshold_artifact_path)

    @property
    def schema_path(self) -> Path:
        return self.resolve_repo_path(self.schema_artifact_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

