#!/usr/bin/env python3
"""Deterministic regression tests for wind_project_model.py."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wind_project_model", ROOT / "scripts" / "wind_project_model.py")
model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(model)
BASE = json.loads((ROOT / "templates" / "project-input.json").read_text(encoding="utf-8"))


def test_template_runs() -> None:
    warnings = model.validate(BASE)
    p50 = model.run_scenario(copy.deepcopy(BASE), "P50")
    p90 = model.run_scenario(copy.deepcopy(BASE), "P90")
    assert p50["project_irr"] is not None
    assert p50["minimum_dscr"] is not None
    assert p90["hours_after_loss_logic"] == 2000
    assert any("ignored" in w for w in warnings)
    assert p90["project_irr"] < p50["project_irr"]


def test_no_double_deduction() -> None:
    d = copy.deepcopy(BASE)
    d["generation"]["hours_mode"] = "net_settlement_hours"
    s = model.run_scenario(d, "P50")
    assert s["hours_after_loss_logic"] == d["generation"]["p50_hours"]


def test_gross_losses_apply() -> None:
    d = copy.deepcopy(BASE)
    d["generation"]["hours_mode"] = "gross_resource_hours"
    d["generation"]["losses"] = {"a": 0.10, "b": 0.20}
    s = model.run_scenario(d, "P50")
    expected = d["generation"]["p50_hours"] * 0.9 * 0.8
    assert abs(s["hours_after_loss_logic"] - expected) < 1e-9


def test_p90_cannot_exceed_p50() -> None:
    d = copy.deepcopy(BASE)
    d["generation"]["p90_hours"] = d["generation"]["p50_hours"] + 1
    try:
        model.validate(d)
    except model.InputError:
        return
    raise AssertionError("invalid P90/P50 relation was accepted")


def test_dscr_block() -> None:
    d = copy.deepcopy(BASE)
    d["project"]["capex"] = 1_200_000_000
    d["financing"]["debt_ratio"] = 0.70
    d["pricing"]["market_capture_price"] = 0.25
    d["pricing"]["mechanism_reference_price"] = 0.25
    d["pricing"]["mechanism_price"] = 0.30
    d["generation"]["p90_hours"] = 2000
    d["costs"]["fixed_annual_cost"] = 9_000_000
    s = model.run_scenario(d, "P90")
    assert s["first_dscr_below_1"] is not None
    assert any(f["code"] == "DSCR_BELOW_1" for f in s["flags"])


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS total={len(tests)}")


if __name__ == "__main__":
    main()
