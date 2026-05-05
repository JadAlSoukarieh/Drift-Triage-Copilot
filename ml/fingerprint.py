from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_pip_freeze_hash() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return sha256_text(result.stdout)


def create_environment_fingerprint() -> dict:
    direct_package_versions = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    return {
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "pip_freeze_hash": get_pip_freeze_hash(),
        "package_versions_used_by_training_script": direct_package_versions,
    }


def save_environment_fingerprint(path: Path) -> dict:
    fingerprint = create_environment_fingerprint()
    path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return fingerprint


def save_model_hash(model_path: Path, output_path: Path) -> str:
    model_hash = sha256_file(model_path)
    output_path.write_text(f"{model_hash}\n", encoding="utf-8")
    return model_hash

