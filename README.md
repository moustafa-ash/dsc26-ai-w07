# W7 task · Nour Pharmacy: serve the shortage model safely

DSC Summer Internship 2026 · AI & Deep Tech track · Week 7 · expected time 3–4 hours

## Scenario

The shortage model works in a notebook. Dr. Mona's IT lead will only let it near the branches
if three things are true: it never leaks patient data, it never leaves a pharmacist without an
answer, and someone will notice when the world changes under it. You will build the service
around a model you are given: validated requests, PII redaction, fallbacks, health and metrics
endpoints, and a drift report.

## Set up

1. Open this starter (`dsc26-ai-w07-starter`), **Use this template → Create a new repository**,
   owner your own account, name exactly **`dsc26-ai-w07`**.
2. Clone, `uv sync`, `uv run pytest` (the public tests fail until you write the code).

Versions: Python 3.12, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, pydantic
2.13.5, fastapi 0.141.1, httpx 0.28.1 (for `TestClient`), pytest 9.1.1. Nothing else: the
grader has nothing else installed.

## Layout

```
dsc26-ai-w07/
├── pyproject.toml         # provided, do not change the dependencies
├── nour/
│   ├── __init__.py
│   ├── fallback.py          # provided: rule_based_risk(...) (do not edit)
│   ├── app.py                # provided: builds a runnable demo app for `fastapi dev` (do not edit)
│   ├── drift.py               # you write
│   ├── guards.py                # you write
│   └── service.py                # you write
├── data/reference.csv, data/current.csv   # provided: one month before and after a supply shock
├── tests/test_public.py
├── DRIFT.md                  # you write
└── AI_USAGE.md
```

The grader restores `nour/fallback.py` and `nour/app.py` before running, so edits there have no
effect on your score.

## What to build

### `nour/drift.py`

```python
def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float: ...
def categorical_psi(reference, current, eps: float = 1e-4) -> float: ...
def drift_report(ref_df, cur_df, numeric: list[str], categorical: list[str], threshold: float = 0.2) -> dict: ...
```

- `psi`: drop NaNs from both; `ValueError` if either is then empty. Interior edges are
  `np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1))[1:-1])`; a value's bin is
  `np.searchsorted(edges, x, side="right")`. Proportions per bin, each floored at `eps`, then
  `Σ (c − r) × ln(c / r)` over the bins.
- `categorical_psi`: drop missing values, compare as strings; categories are the union of both
  sides (so a category that is new in `current` still counts, with proportion 0 floored to
  `eps` on the reference side — this is what makes a brand-new category show up as a large PSI,
  not an error). Same flooring and formula as `psi`.
- `drift_report`: returns `{"columns": {col: {"psi", "missing_rate_ref", "missing_rate_cur",
  "drifted"}}, "drifted_columns": [...]}` where `drifted` is true when `psi >= threshold` or the
  missing rate changed by 0.1 or more (absolute), and `drifted_columns` is sorted.

### `nour/guards.py`

```python
def find_pii(text: str) -> list[dict]: ...     # [{"type": "NATIONAL_ID" | "PHONE", "start": int, "end": int}]
def redact(text: str) -> tuple[str, list[dict]]: ...
```

- Digits may be ASCII, Arabic-Indic (٠–٩) or Persian (۰–۹), in any mix within one match.
- `NATIONAL_ID`: exactly 14 consecutive digits not touching other digits on either side; first
  digit 2 or 3 (2 → born 19xx, 3 → born 20xx); digits 2–7 a real calendar date YYMMDD in that
  century (so day 31 in a 30-day month, 29 February in a non-leap year, or month 13 all fail);
  digits 8–9 a known governorate code: `01`–`04`, `11`–`19`, `21`–`29`, `31`–`35`, `88`. The
  14th digit (the checksum) is **not** checked.
- `PHONE`: an Egyptian mobile number — `0`, then `1`, then one of `0 1 2 5`, then eight more
  digits — optionally preceded by a country prefix, which is exactly one of `+2`, `002` or `2`
  (so `+201012345678` and `00201012345678` both match, as does the bare `201012345678`). Single
  spaces or dashes may separate digit groups anywhere in the number, including inside the
  prefix. The match must not be preceded by a digit or a `+` that is not itself part of the
  match, and must not be followed by a digit (so `013...` never matches, and an 11-digit or
  13-digit run of otherwise-valid digits is rejected at the boundary).
- A national ID match wins over an overlapping phone match. Offsets refer to the **original**
  text (not the ASCII-digit-translated copy you may build internally). Results sorted by
  `start`.
- `redact` replaces each match with `[NATIONAL_ID]` or `[PHONE]` and returns the new text and
  the findings. Text without PII comes back unchanged.

### `nour/service.py`

```python
class PredictRequest(BaseModel): ...
class PredictResponse(BaseModel): ...
def band_for(risk: float) -> str: ...
class ShortageService:
    def __init__(self, model, model_version: str, training_ranges: dict[str, tuple[float, float]]): ...
    log: list[dict]
    def predict(self, req: PredictRequest) -> PredictResponse: ...
    def metrics(self) -> dict: ...
def create_app(service: ShortageService) -> FastAPI: ...
```

- `PredictRequest` (unknown fields rejected): `branch_id` matching `^BR-\d{3}$`; `item_code`
  matching `^[A-Z]{2}-\d{5}$`; `weeks_of_cover` float 0–52; `supplier_lead_time_days` int 0–180;
  `orders_last_4w` int ≥ 0; `supplier_fill_rate` float 0–1; `pharmacist_note` str, default `""`,
  at most 500 characters.
- `PredictResponse`: `risk` (float), `band` (`low` / `medium` / `high`), `source` (`model` /
  `fallback`), `note_redacted` (str), `model_version` (str).
- `band_for`: below 0.3 is `low`, below 0.7 is `medium`, otherwise `high` (so 0.3 is medium and
  0.7 is high).
- `predict`: if any feature named in `training_ranges` is outside its inclusive range, use the
  fallback (reason `out_of_range`) without calling the model. Otherwise call
  `model.predict_proba` with a 2-D numpy array of exactly one row in this column order:
  `weeks_of_cover`, `supplier_lead_time_days`, `orders_last_4w`, `supplier_fill_rate`, and take
  column 1. If the call raises (reason `model_error`), or the value is not finite or not within
  `[0, 1]` (reason `invalid_output`), use `rule_based_risk`. Always redact the note, whichever
  path answered.
- Every call appends one entry to `service.log` with at least `branch_id`, `item_code`, `risk`,
  `source`, `reason` (`None` when the model answered), `note` (the **redacted** note only) and
  `latency_ms`. Nothing in the log may contain the raw note.
- `metrics()`: `{"requests", "fallbacks", "fallback_rate", "p95_latency_ms"}` from the log
  (`fallback_rate` 0.0 and `p95_latency_ms` 0.0 when empty; p95 via `np.percentile(..., 95)`).
- `create_app(service)`: `POST /predict` (request and response models above; invalid requests
  get FastAPI's 422), `GET /health` → `{"status": "ok", "model_version": ...}`, `GET /metrics` →
  `service.metrics()`. No endpoint may return a traceback.

### `DRIFT.md`

Run `drift_report` on the two provided CSVs (numeric: `weeks_of_cover`,
`supplier_lead_time_days`, `orders_last_4w`, `supplier_fill_rate`; categorical:
`therapeutic_class`), paste the table, and write five sentences for Dr. Mona: what moved, what
it probably means, and what the service should do. Read by your Track Mentor; not graded by
tests.

## `AI_USAGE.md`

Tools and what for; the PII cases your AI-written regex missed at first and how you found them;
one sentence on what your service does when the model is down, written for a pharmacist. Read
by your Track Mentor, not graded by tests.

## How grading works

Push, then **Submit for grading** on DSC Hex. The grader clones that commit, restores
`nour/fallback.py` and `nour/app.py`, and runs the hidden tests on Python 3.12 with the versions
above, network off, 60 seconds per test, 5 minutes in all. Score 0–100 with feedback per test
name; each test is all or nothing; up to five runs per task per day; the best attempt counts.

| Test name | What it checks | Points |
|---|---|---|
| `psi_matches_reference` | Property: 30 random pairs (sizes, shapes, `n_bins`) against the reference formula within 1e-9 | 12 |
| `psi_behaviour` | Same distribution near 0; large shift above 0.25; symmetric inputs give non-negative values | 5 |
| `psi_edge_cases` | Constant reference, NaNs ignored, all-NaN or empty side raises | 6 |
| `categorical_psi_new_category` | A category absent from the reference raises PSI; missing values ignored | 5 |
| `drift_report_flags` | Right columns flagged by PSI and by missing-rate jump; sorted list; threshold respected | 6 |
| `national_id_valid_and_invalid` | Valid IDs in both centuries and all digit scripts; bad month, day 31 in a 30-day month, 29 Feb in a non-leap year, unknown governorate, 13 and 15 digits not matched | 10 |
| `phone_formats` | Local, `+20`, `0020`, spaces, dashes, Arabic-Indic and mixed digits; `013…`, too short, too long not matched | 8 |
| `redaction_offsets_and_bystanders` | Offsets point at the original text; prices, dates, item codes and 10-digit numbers untouched; text without PII unchanged | 6 |
| `request_validation` | 422 for bad branch, bad item code, negative values, fill rate above 1, extra field, 501-character note | 6 |
| `model_error_falls_back` | Raising model: `fallback`, reason `model_error`, risk equals `rule_based_risk` | 6 |
| `invalid_output_falls_back` | NaN, infinity, 1.2 and −0.1 outputs all fall back with reason `invalid_output` | 5 |
| `out_of_range_skips_the_model` | Out-of-range input never calls the model; inclusive boundaries call it | 5 |
| `band_boundaries` | 0.2999, 0.3, 0.6999, 0.7 | 4 |
| `logs_hold_no_raw_pii` | Ten requests with planted IDs and phones: no raw value anywhere in `service.log` or any response | 8 |
| `health_and_metrics` | `/health` shape and version; `/metrics` counts, fallback rate, p95 present and non-negative | 8 |

Late policy: ×1.0 within 72 hours of the session starting, ×0.6 in the following 7 days, ×0.5
after that until the W8 deadline. During W5 (consolidation week) you can resubmit at ×1.0.

If you are stuck for more than an hour, ask in the AI track channel on Discord or ask Hexa
rather than spending grader runs.
