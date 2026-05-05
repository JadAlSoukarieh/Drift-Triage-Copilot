from __future__ import annotations

from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict

API_TO_MODEL_COLUMN_MAP = {
    "emp_var_rate": "emp.var.rate",
    "cons_price_idx": "cons.price.idx",
    "cons_conf_idx": "cons.conf.idx",
    "nr_employed": "nr.employed",
}

TRAINING_FEATURE_ORDER = [
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


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

    training_feature_order: ClassVar[list[str]] = TRAINING_FEATURE_ORDER

    def to_model_dataframe(self) -> pd.DataFrame:
        payload = self.model_dump()
        model_payload = {
            "age": payload["age"],
            "job": payload["job"],
            "marital": payload["marital"],
            "education": payload["education"],
            "default": payload["default"],
            "housing": payload["housing"],
            "loan": payload["loan"],
            "contact": payload["contact"],
            "month": payload["month"],
            "day_of_week": payload["day_of_week"],
            "campaign": payload["campaign"],
            "pdays": payload["pdays"],
            "previous": payload["previous"],
            "poutcome": payload["poutcome"],
            "emp.var.rate": payload["emp_var_rate"],
            "cons.price.idx": payload["cons_price_idx"],
            "cons.conf.idx": payload["cons_conf_idx"],
            "euribor3m": payload["euribor3m"],
            "nr.employed": payload["nr_employed"],
            "pdays_was_999": int(payload["pdays"] == 999),
        }
        frame = pd.DataFrame([model_payload])
        return frame[self.training_feature_order]


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_alias: str
    prediction: int
    label: str
    probability: float
    threshold: float
    created_at: str

