"""The shortage-risk service: validated requests, guarded prediction, health and metrics.

You write this file.
"""
from __future__ import annotations

from math import isfinite
from time import perf_counter
from typing import Literal

import numpy as np
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nour.fallback import rule_based_risk
from nour.guards import redact


_FEATURES = ("weeks_of_cover", "supplier_lead_time_days", "orders_last_4w", "supplier_fill_rate")


class PredictRequest(BaseModel):
    """You fill in the fields. Unknown fields must be rejected (`extra="forbid"`). See
    README.md for every field's exact type and bounds: branch_id (^BR-\\d{3}$), item_code
    (^[A-Z]{2}-\\d{5}$), weeks_of_cover (0-52), supplier_lead_time_days (0-180 int),
    orders_last_4w (>=0 int), supplier_fill_rate (0-1), pharmacist_note (default "", at most
    500 characters). Until you do, every request is rejected with a validation error.
    """
    model_config = {"extra": "forbid"}

    branch_id: str = Field(pattern=r"^BR-\d{3}$")
    item_code: str = Field(pattern=r"^[A-Z]{2}-\d{5}$")
    weeks_of_cover: float = Field(ge=0, le=52)
    supplier_lead_time_days: int = Field(ge=0, le=180)
    orders_last_4w: int = Field(ge=0)
    supplier_fill_rate: float = Field(ge=0, le=1)
    pharmacist_note: str = Field(default="", max_length=500)

class PredictResponse(BaseModel):
    """You fill in the fields: risk: float; band: "low"/"medium"/"high";
    source: "model"/"fallback"; note_redacted: str; model_version: str."""

    risk: float
    band: Literal["low", "medium", "high"]
    source: Literal["model", "fallback"]
    note_redacted: str
    model_version: str


def band_for(risk: float) -> Literal["low", "medium", "high"]:
    """Below 0.3 -> "low"; below 0.7 -> "medium"; otherwise "high" (so 0.3 is medium, 0.7 is high)."""
    if risk < 0.3:
        return "low"
    if risk < 0.7:
        return "medium"
    return "high"


class ShortageService:
    def __init__(self, model, model_version: str, training_ranges: dict[str, tuple[float, float]]):
        """`training_ranges` maps a subset of the model's feature names to an inclusive
        (lo, hi) range. `self.log: list[dict]` accumulates one entry per `predict` call."""
        self.model = model
        self.model_version = model_version
        self.training_ranges = dict(training_ranges)
        self.log: list[dict] = []

    def predict(self, req: PredictRequest) -> PredictResponse:
        """See README.md "What to build -> nour/service.py" for the exact fallback rules
        (out_of_range, model_error, invalid_output), the feature column order passed to
        `model.predict_proba`, the redaction step, and the log entry's shape."""
        start = perf_counter()
        note, _ = redact(req.pharmacist_note)
        risk, reason = self._model_risk(req)
        source = "model" if reason is None else "fallback"
        if reason is not None:
            risk = rule_based_risk(req.weeks_of_cover, req.supplier_fill_rate, req.supplier_lead_time_days)
        response = PredictResponse(
            risk=risk, band=band_for(risk), source=source, note_redacted=note,
            model_version=self.model_version,
        )
        self.log.append({
            "branch_id": req.branch_id, "item_code": req.item_code, "risk": risk,
            "source": source, "reason": reason, "note": note,
            "latency_ms": (perf_counter() - start) * 1000,
        })
        return response

    def _model_risk(self, req: PredictRequest) -> tuple[float, str | None]:
        if any(not low <= getattr(req, field) <= high
               for field, (low, high) in self.training_ranges.items()):
            return 0.0, "out_of_range"
        try:
            inputs = np.array([[getattr(req, name) for name in _FEATURES]], dtype=float)
            probabilities = self.model.predict_proba(inputs)
        except Exception:
            return 0.0, "model_error"
        try:
            risk = float(np.asarray(probabilities)[0, 1])
        except (TypeError, ValueError, IndexError, OverflowError):
            return 0.0, "invalid_output"
        if not isfinite(risk) or not 0 <= risk <= 1:
            return 0.0, "invalid_output"
        return risk, None

    def metrics(self) -> dict:
        """{"requests", "fallbacks", "fallback_rate", "p95_latency_ms"} computed from self.log
        (zeros when the log is empty; p95 via numpy.percentile(..., 95))."""
        entries = tuple(self.log)
        requests = len(entries)
        fallbacks = sum(entry["source"] == "fallback" for entry in entries)
        return {
            "requests": requests,
            "fallbacks": fallbacks,
            "fallback_rate": fallbacks / requests if requests else 0.0,
            "p95_latency_ms": float(np.percentile([entry["latency_ms"] for entry in entries], 95))
            if requests else 0.0,
        }


def create_app(service: ShortageService) -> FastAPI:
    """POST /predict, GET /health -> {"status": "ok", "model_version": ...}, GET /metrics.
    No endpoint may return a traceback."""
    app = FastAPI(debug=False)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_, exc: RequestValidationError):
        # Error inputs and unknown field names can contain text supplied by the caller.
        details = [
            {
                "type": error["type"],
                "loc": ("body",) if error["type"] == "extra_forbidden" else error["loc"],
                "msg": error["msg"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        return service.predict(req)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": service.model_version}

    @app.get("/metrics")
    def metrics() -> dict:
        return service.metrics()

    return app
