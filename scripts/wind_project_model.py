#!/usr/bin/env python3
"""Deterministic wind-project screening model (stdlib only)."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


class InputError(ValueError):
    pass


def require(d: dict, path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise InputError(f"missing required field: {path}")
        cur = cur[key]
    return cur


def number(d: dict, path: str, *, low: float | None = None, high: float | None = None) -> float:
    value = require(d, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError(f"{path} must be a finite number")
    value = float(value)
    if low is not None and value < low:
        raise InputError(f"{path} must be >= {low}")
    if high is not None and value > high:
        raise InputError(f"{path} must be <= {high}")
    return value


def validate(d: dict) -> list[str]:
    number(d, "project.capacity_mw", low=0.001)
    number(d, "project.operating_years", low=1)
    number(d, "project.capex", low=0.01)
    number(d, "generation.p50_hours", low=0, high=8760)
    number(d, "generation.p90_hours", low=0, high=8760)
    number(d, "generation.annual_degradation", low=0, high=0.2)
    number(d, "pricing.market_capture_price", low=-5, high=10)
    number(d, "pricing.mechanism_share", low=0, high=1)
    number(d, "pricing.mechanism_price", low=-5, high=10)
    number(d, "pricing.mechanism_reference_price", low=-5, high=10)
    number(d, "pricing.mechanism_years", low=0)
    number(d, "costs.opex_yuan_per_kw_year", low=0)
    number(d, "costs.opex_escalation", low=-0.5, high=1)
    number(d, "tax.standard_rate", low=0, high=1)
    number(d, "financing.debt_ratio", low=0, high=1)
    number(d, "financing.interest_rate", low=0, high=1)
    number(d, "financing.tenor_years", low=1)
    number(d, "financing.minimum_dscr", low=0)
    number(d, "valuation.wacc", low=0, high=1)
    mode = require(d, "generation.hours_mode")
    if mode not in {"net_settlement_hours", "gross_resource_hours"}:
        raise InputError("generation.hours_mode must be net_settlement_hours or gross_resource_hours")
    repayment = require(d, "financing.repayment")
    if repayment not in {"equal_principal", "annuity"}:
        raise InputError("financing.repayment must be equal_principal or annuity")
    if number(d, "generation.p90_hours") > number(d, "generation.p50_hours"):
        raise InputError("P90 hours cannot exceed P50 hours")
    if int(number(d, "financing.tenor_years")) > int(number(d, "project.operating_years")):
        raise InputError("debt tenor cannot exceed operating years")
    warnings: list[str] = []
    evidence = d.get("evidence", {})
    for key in ("generation", "pricing", "capex", "financing", "tax"):
        if not evidence.get(key):
            warnings.append(f"missing evidence label: {key}")
    if mode == "net_settlement_hours" and any(float(v) != 0 for v in d.get("generation", {}).get("losses", {}).values()):
        warnings.append("losses were supplied but ignored because hours_mode=net_settlement_hours; this prevents double deduction")
    return warnings


def npv(rate: float, cashflows: list[float]) -> float:
    if rate <= -1:
        return math.inf
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def irr(cashflows: list[float]) -> float | None:
    if not any(x < 0 for x in cashflows) or not any(x > 0 for x in cashflows):
        return None
    # Scan log(1+r) so both negative and very high rates are covered.
    points = [-0.999] + [math.exp(-6 + i * 0.025) - 1 for i in range(0, 520)]
    previous_r, previous_v = points[0], npv(points[0], cashflows)
    for r in points[1:]:
        value = npv(r, cashflows)
        if math.isfinite(value) and math.isfinite(previous_v) and value * previous_v <= 0:
            lo, hi = previous_r, r
            vlo = previous_v
            for _ in range(160):
                mid = (lo + hi) / 2
                vmid = npv(mid, cashflows)
                if abs(vmid) < 1e-7:
                    return mid
                if vmid * vlo <= 0:
                    hi = mid
                else:
                    lo, vlo = mid, vmid
            return (lo + hi) / 2
        previous_r, previous_v = r, value
    return None


def tax_rate_for_year(tax: dict, year: int) -> float:
    standard = float(tax["standard_rate"])
    holiday = int(tax.get("holiday_years", 0))
    half = int(tax.get("half_rate_years", 0))
    if year <= holiday:
        return 0.0
    if year <= holiday + half:
        return standard / 2
    return standard


def debt_schedule(d: dict) -> list[dict[str, float]]:
    capex = float(d["project"]["capex"])
    f = d["financing"]
    original = capex * float(f["debt_ratio"])
    rate = float(f["interest_rate"])
    tenor = int(f["tenor_years"])
    grace = int(f.get("grace_years", 0))
    amort_years = tenor - grace
    if amort_years <= 0:
        raise InputError("grace_years must be less than tenor_years")
    balance = original
    result = []
    annuity = 0.0
    if f["repayment"] == "annuity":
        annuity = original / amort_years if rate == 0 else original * rate / (1 - (1 + rate) ** (-amort_years))
    for year in range(1, int(d["project"]["operating_years"]) + 1):
        opening = balance
        interest = opening * rate if year <= tenor else 0.0
        if year > tenor:
            principal = 0.0
        elif year <= grace:
            principal = 0.0
        elif f["repayment"] == "equal_principal":
            principal = min(opening, original / amort_years)
        else:
            principal = min(opening, max(0.0, annuity - interest))
        balance = max(0.0, opening - principal)
        result.append({"year": year, "opening": opening, "interest": interest, "principal": principal,
                       "debt_service": interest + principal, "closing": balance})
    return result


def net_hours(d: dict, hours: float) -> float:
    if d["generation"]["hours_mode"] == "net_settlement_hours":
        return hours
    factor = 1.0
    for value in d["generation"].get("losses", {}).values():
        loss = float(value)
        if loss < 0 or loss >= 1:
            raise InputError("each generation loss must be in [0,1)")
        factor *= 1 - loss
    return hours * factor


def run_scenario(d: dict, scenario: str, hours_override: float | None = None,
                 mechanism_price_override: float | None = None) -> dict[str, Any]:
    years = int(d["project"]["operating_years"])
    capacity_kw = float(d["project"]["capacity_mw"]) * 1000
    capex = float(d["project"]["capex"])
    dep_fraction = float(d["project"].get("depreciable_fraction", 1.0))
    depreciation = capex * dep_fraction / years
    salvage = float(d["project"].get("salvage_value", 0))
    base_hours = hours_override if hours_override is not None else float(d["generation"][f"{scenario.lower()}_hours"])
    base_net_hours = net_hours(d, base_hours)
    degradation = float(d["generation"]["annual_degradation"])
    price = d["pricing"]
    market = float(price["market_capture_price"])
    mech_price = float(mechanism_price_override if mechanism_price_override is not None else price["mechanism_price"])
    mech_ref = float(price["mechanism_reference_price"])
    mech_share = float(price["mechanism_share"])
    mech_years = int(price["mechanism_years"])
    green = float(price.get("green_value_per_kwh", 0))
    subsidy = float(price.get("subsidy_per_kwh", 0))
    other_revenue = float(price.get("other_annual_revenue", 0))
    costs = d["costs"]
    base_opex = capacity_kw * float(costs["opex_yuan_per_kw_year"]) + float(costs.get("fixed_annual_cost", 0))
    escalation = float(costs["opex_escalation"])
    wc_change = float(costs.get("working_capital_change", 0))
    repairs = {int(k): float(v) for k, v in costs.get("major_repairs", {}).items()}
    debt = debt_schedule(d)
    debt_draw = capex * float(d["financing"]["debt_ratio"])
    equity = capex - debt_draw
    rows: list[dict[str, Any]] = []
    project_cf = [-capex]
    equity_cf = [-equity]
    for year in range(1, years + 1):
        hours = base_net_hours * ((1 - degradation) ** (year - 1))
        generation_kwh = capacity_kw * hours
        market_revenue = generation_kwh * market
        mechanism_delta = generation_kwh * mech_share * (mech_price - mech_ref) if year <= mech_years else 0.0
        revenue = market_revenue + mechanism_delta + generation_kwh * (green + subsidy) + other_revenue
        opex = base_opex * ((1 + escalation) ** (year - 1))
        ebitda = revenue - opex
        interest = debt[year - 1]["interest"]
        rate = tax_rate_for_year(d["tax"], year)
        project_tax = max(0.0, ebitda - depreciation) * rate
        cash_tax = max(0.0, ebitda - depreciation - interest) * rate
        maintenance_capex = repairs.get(year, 0.0)
        project_fcf = ebitda - project_tax - wc_change - maintenance_capex
        cfads = ebitda - cash_tax - wc_change - maintenance_capex
        debt_service = debt[year - 1]["debt_service"]
        dscr = cfads / debt_service if debt_service > 0 else None
        equity_cash = cfads - debt_service
        if year == years:
            project_fcf += salvage
            equity_cash += salvage
        project_cf.append(project_fcf)
        equity_cf.append(equity_cash)
        rows.append({
            "year": year, "net_hours": hours, "generation_kwh": generation_kwh,
            "market_revenue": market_revenue, "mechanism_delta": mechanism_delta,
            "revenue": revenue, "opex": opex, "ebitda": ebitda,
            "project_tax": project_tax, "cash_tax": cash_tax,
            "maintenance_capex": maintenance_capex, "cfads": cfads,
            **debt[year - 1], "dscr": dscr, "project_fcf": project_fcf,
            "equity_cashflow": equity_cash
        })
    dscrs = [r["dscr"] for r in rows if r["dscr"] is not None]
    min_dscr = min(dscrs) if dscrs else None
    first_below_one = next((r["year"] for r in rows if r["dscr"] is not None and r["dscr"] < 1), None)
    threshold = float(d["financing"]["minimum_dscr"])
    first_below_threshold = next((r["year"] for r in rows if r["dscr"] is not None and r["dscr"] < threshold), None)
    project_irr = irr(project_cf)
    equity_irr = irr(equity_cf)
    wacc = float(d["valuation"]["wacc"])
    flags = []
    if first_below_one is not None:
        flags.append({"severity": "block", "code": "DSCR_BELOW_1", "year": first_below_one})
    elif first_below_threshold is not None:
        flags.append({"severity": "high", "code": "DSCR_BELOW_THRESHOLD", "year": first_below_threshold,
                      "threshold": threshold})
    if project_irr is None:
        flags.append({"severity": "high", "code": "PROJECT_IRR_UNAVAILABLE"})
    elif project_irr <= wacc:
        flags.append({"severity": "high", "code": "PROJECT_IRR_NOT_ABOVE_WACC", "wacc": wacc})
    return {
        "scenario": scenario.upper(), "hours_input": base_hours, "hours_after_loss_logic": base_net_hours,
        "project_irr": project_irr, "equity_irr": equity_irr, "project_npv_at_wacc": npv(wacc, project_cf),
        "minimum_dscr": min_dscr, "first_dscr_below_1": first_below_one,
        "first_dscr_below_threshold": first_below_threshold, "flags": flags,
        "cashflows": {"project": project_cf, "equity": equity_cf}, "annual": rows
    }


def summarize_sensitivity(d: dict) -> list[dict[str, Any]]:
    result = []
    for hours in d.get("sensitivity", {}).get("hours", []):
        for price in d.get("sensitivity", {}).get("mechanism_prices", []):
            scenario = run_scenario(d, "P50", float(hours), float(price))
            result.append({"hours": float(hours), "mechanism_price": float(price),
                           "project_irr": scenario["project_irr"], "equity_irr": scenario["equity_irr"],
                           "minimum_dscr": scenario["minimum_dscr"],
                           "first_dscr_below_1": scenario["first_dscr_below_1"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    try:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        warnings = validate(data)
        p50 = run_scenario(copy.deepcopy(data), "P50")
        p90 = run_scenario(copy.deepcopy(data), "P90")
        result = {
            "model": "wind-project-screening-v0.1.0", "units": {"money": "CNY", "energy": "kWh", "price": "CNY/kWh"},
            "warnings": warnings, "evidence": data.get("evidence", {}),
            "scenarios": {"P50": p50, "P90": p90}, "sensitivity": summarize_sensitivity(data),
            "decision": {
                "p90_blocked": any(f["severity"] == "block" for f in p90["flags"]),
                "p90_flags": p90["flags"],
                "note": "Deterministic screening only; verify technical, legal, tax and market assumptions before approval."
            }
        }
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2), encoding="utf-8")
        print(json.dumps({"status": "success", "output": str(Path(args.output).resolve()),
                          "p50_project_irr": p50["project_irr"], "p50_min_dscr": p50["minimum_dscr"],
                          "p90_project_irr": p90["project_irr"], "p90_min_dscr": p90["minimum_dscr"],
                          "p90_blocked": result["decision"]["p90_blocked"], "warnings": len(warnings)}, ensure_ascii=False))
        return 0
    except (OSError, json.JSONDecodeError, InputError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
