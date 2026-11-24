# Drift report

Computed from the supplied reference and current CSVs (3,000 rows each). Flag thresholds are PSI of at least 0.2 or a change in the missing rate of at least 10 percentage points.

| column | PSI | missing (ref) | missing (current) | drifted |
|---|---|---|---|---|
| `weeks_of_cover` | 1.2173 | 0.00% | 0.00% | Yes |
| `supplier_lead_time_days` | 3.3172 | 0.00% | 0.00% | Yes |
| `orders_last_4w` | 0.0037 | 0.00% | 0.00% | No |
| `supplier_fill_rate` | 3.7145 | 3.10% | 15.67% | Yes |
| `therapeutic_class` | 0.0023 | 0.00% | 0.00% | No |

Dr. Mona, median stock cover fell from eight to five weeks while median supplier lead time rose from 14 to 26 days.
Median reported fill rate fell from 92% to 73.8%, and more rows now lack a fill rate.
Order volumes and the mix of medicine classes were comparatively stable, which suggests a supplier-side disruption and warrants checking the fill-rate reporting feed.
The service should keep labelling rule-based answers as fallback and alert the team when these shifts or missing-data rates exceed the agreed thresholds.
Review recent shortages and model accuracy before updating the model or widening the feature ranges.
