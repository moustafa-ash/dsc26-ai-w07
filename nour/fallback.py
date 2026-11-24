"""Rule-based shortage risk. Provided by DSC. Do not edit."""
from __future__ import annotations


def rule_based_risk(weeks_of_cover: float, supplier_fill_rate: float, supplier_lead_time_days: int) -> float:
    """A transparent baseline the pharmacists already understand."""
    risk = 0.15
    if weeks_of_cover < 2:
        risk += 0.35
    elif weeks_of_cover < 4:
        risk += 0.15
    if supplier_fill_rate < 0.7:
        risk += 0.3
    if supplier_lead_time_days > 21:
        risk += 0.15
    return min(risk, 0.95)
