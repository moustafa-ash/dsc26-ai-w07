import numpy as np
import pytest
from fastapi.testclient import TestClient

from nour.drift import psi
from nour.guards import redact
from nour.service import PredictRequest, ShortageService, create_app

RANGES = {"weeks_of_cover": (0, 26), "supplier_lead_time_days": (0, 90)}
GOOD = {"branch_id": "BR-014", "item_code": "NP-10423", "weeks_of_cover": 1.5,
        "supplier_lead_time_days": 30, "orders_last_4w": 120, "supplier_fill_rate": 0.6,
        "pharmacist_note": "patient asked again, call 01012345678"}


class ConstantModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]] * len(X))


class BrokenModel:
    def predict_proba(self, X):
        raise RuntimeError("model file corrupted")


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.25


def test_redact_hides_phone_and_national_id():
    text, found = redact("المريض رقمه ٠١٠١٢٣٤٥٦٧٨ والرقم القومي 29001011234567")
    assert "[PHONE]" in text and "[NATIONAL_ID]" in text
    assert [f["type"] for f in found] == ["PHONE", "NATIONAL_ID"]


def test_predict_uses_the_model_and_redacts_the_note():
    client = TestClient(create_app(ShortageService(ConstantModel(0.8), "v1", RANGES)))
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "model" and body["band"] == "high"
    assert "01012345678" not in body["note_redacted"]


def test_broken_model_falls_back_instead_of_failing():
    service = ShortageService(BrokenModel(), "v1", RANGES)
    out = service.predict(PredictRequest(**GOOD))
    assert out.source == "fallback"
    assert 0.0 <= out.risk <= 1.0


def test_bad_request_is_rejected_with_422():
    client = TestClient(create_app(ShortageService(ConstantModel(0.2), "v1", RANGES)))
    assert client.post("/predict", json={**GOOD, "branch_id": "14"}).status_code == 422
