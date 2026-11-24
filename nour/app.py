"""A runnable demo app, wired to a small throwaway model. Provided by DSC. Do not edit.

    fastapi dev nour/app.py

This is for your own manual testing (curl, the FastAPI docs UI); it is never imported by any
test. The demo model is a plain `LogisticRegression` fit on synthetic data at import time, so
`app.py` has nothing to load from disk and nothing network-dependent.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from nour.service import ShortageService, create_app

TRAINING_RANGES = {
    "weeks_of_cover": (0.0, 26.0),
    "supplier_lead_time_days": (0, 90),
    "orders_last_4w": (0, 500),
    "supplier_fill_rate": (0.0, 1.0),
}


def _demo_model():
    rng = np.random.default_rng(7)
    n = 2000
    weeks = rng.uniform(0, 26, n)
    lead = rng.integers(0, 90, n)
    orders = rng.integers(0, 500, n)
    fill = rng.uniform(0, 1, n)
    risk_signal = (weeks < 3) * 0.4 + (fill < 0.6) * 0.3 + (lead > 30) * 0.2
    y = (risk_signal + rng.normal(0, 0.1, n)) > 0.35
    x = np.column_stack([weeks, lead, orders, fill])
    model = LogisticRegression()
    model.fit(x, y)
    return model


service = ShortageService(_demo_model(), "demo-0.1", TRAINING_RANGES)
app = create_app(service)
