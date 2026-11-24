"""The shortage-risk service: validated requests, guarded prediction, health and metrics.

You write this file.
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """You fill in the fields. Unknown fields must be rejected (`extra="forbid"`). See
    README.md for every field's exact type and bounds: branch_id (^BR-\\d{3}$), item_code
    (^[A-Z]{2}-\\d{5}$), weeks_of_cover (0-52), supplier_lead_time_days (0-180 int),
    orders_last_4w (>=0 int), supplier_fill_rate (0-1), pharmacist_note (default "", at most
    500 characters). Until you do, every request is rejected with a validation error.
    """
    model_config = {"extra": "forbid"}


class PredictResponse(BaseModel):
    """You fill in the fields: risk: float; band: "low"/"medium"/"high";
    source: "model"/"fallback"; note_redacted: str; model_version: str."""


def band_for(risk: float) -> str:
    """Below 0.3 -> "low"; below 0.7 -> "medium"; otherwise "high" (so 0.3 is medium, 0.7 is high)."""
    raise NotImplementedError


class ShortageService:
    def __init__(self, model, model_version: str, training_ranges: dict[str, tuple[float, float]]):
        """`training_ranges` maps a subset of the model's feature names to an inclusive
        (lo, hi) range. `self.log: list[dict]` accumulates one entry per `predict` call."""
        raise NotImplementedError

    def predict(self, req: PredictRequest) -> PredictResponse:
        """See README.md "What to build -> nour/service.py" for the exact fallback rules
        (out_of_range, model_error, invalid_output), the feature column order passed to
        `model.predict_proba`, the redaction step, and the log entry's shape."""
        raise NotImplementedError

    def metrics(self) -> dict:
        """{"requests", "fallbacks", "fallback_rate", "p95_latency_ms"} computed from self.log
        (zeros when the log is empty; p95 via numpy.percentile(..., 95))."""
        raise NotImplementedError


def create_app(service: ShortageService) -> FastAPI:
    """POST /predict, GET /health -> {"status": "ok", "model_version": ...}, GET /metrics.
    No endpoint may return a traceback."""
    raise NotImplementedError
