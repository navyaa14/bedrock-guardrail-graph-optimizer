"""
test_graph_metrics.py
-----------------------
Tests proving graph_metrics.py enforces the hard safety gates (100%
protected-boundary coverage, 100% high-risk preservation), computes
latency/cost savings correctly against the config-driven savings model,
and that strict mode changes the resulting plan's decision mix.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from synthetic_workflows import generate_synthetic_workflows
from graph_builder import build_all_workflow_graphs, PROTECTED_BOUNDARIES
from graph_features import compute_all_workflow_features
from graph_optimizer import load_policy, generate_optimized_plan
from graph_metrics import compute_metrics, compute_boundary_coverage_matrix, compute_scenario_metrics, compute_scenario_metrics

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "guardrail_policy.yaml")


@pytest.fixture(scope="module")
def policy():
    return load_policy(CONFIG_PATH)


@pytest.fixture(scope="module")
def df():
    return generate_synthetic_workflows(n_workflows=150, seed=9)


@pytest.fixture(scope="module")
def graphs(df):
    return build_all_workflow_graphs(df)


@pytest.fixture(scope="module")
def plan(graphs, policy):
    return generate_optimized_plan(graphs, policy, strict_mode=False)


@pytest.fixture(scope="module")
def feature_df(graphs, policy):
    return compute_all_workflow_features(graphs, policy.get("risk_score_weights", {}))


@pytest.fixture(scope="module")
def metrics(df, graphs, plan, feature_df, policy):
    return compute_metrics(df, graphs, plan, feature_df, policy)


def test_boundary_coverage_is_100_percent(metrics):
    assert metrics["safety"]["boundary_coverage_preserved_percent"] == 100.0


def test_high_risk_preservation_is_100_percent(metrics):
    assert metrics["safety"]["high_risk_preservation_percent"] == 100.0


def test_latency_saved_calculation_is_correct(df, metrics):
    original = float(df["latency_ms"].sum())
    overall = metrics["overall"]
    assert overall["original_total_latency_ms"] == round(original, 2)
    assert overall["optimized_total_latency_ms"] <= overall["original_total_latency_ms"]
    assert overall["latency_saved_ms"] == round(
        overall["original_total_latency_ms"] - overall["optimized_total_latency_ms"], 2
    )


def test_cost_saved_calculation_is_correct(df, metrics):
    original = float(df["estimated_cost_usd"].sum())
    overall = metrics["overall"]
    assert overall["original_total_cost_usd"] == round(original, 4)
    assert overall["optimized_total_cost_usd"] <= overall["original_total_cost_usd"]
    assert overall["cost_saved_usd"] == round(
        overall["original_total_cost_usd"] - overall["optimized_total_cost_usd"], 4
    )


def test_boundary_coverage_matrix_includes_all_protected_boundaries(plan, df, policy):
    matrix = compute_boundary_coverage_matrix(plan, df, policy)
    for boundary in PROTECTED_BOUNDARIES:
        assert boundary in matrix["trust_boundary"].values
        row = matrix[matrix["trust_boundary"] == boundary].iloc[0]
        assert row["coverage_percent"] == 100.0


def test_strict_mode_prevents_skip_redundant_with_audit(graphs, policy):
    strict_plan = generate_optimized_plan(graphs, policy, strict_mode=True)
    assert (strict_plan["recommended_decision"] != "SKIP_REDUNDANT_WITH_AUDIT").all()
    normal_plan = generate_optimized_plan(graphs, policy, strict_mode=False)
    assert (normal_plan["recommended_decision"] == "SKIP_REDUNDANT_WITH_AUDIT").any()


def test_metrics_fail_if_protected_boundary_check_is_skipped(df, graphs, plan, feature_df, policy):
    tampered_plan = plan.copy()
    protected_rows = tampered_plan[tampered_plan["trust_boundary"].isin(PROTECTED_BOUNDARIES)]
    assert len(protected_rows) > 0
    idx = protected_rows.index[0]
    tampered_plan.loc[idx, "recommended_decision"] = "SKIP_REDUNDANT_WITH_AUDIT"

    with pytest.raises(RuntimeError):
        compute_metrics(df, graphs, tampered_plan, feature_df, policy)


def test_compute_scenario_metrics_returns_one_row_per_scenario(df, plan, policy):
    scenario_df = compute_scenario_metrics(df, plan, policy)
    assert len(scenario_df) > 0
    assert set(scenario_df["scenario"]) == set(df["workflow_template"].unique())
    # No duplicate scenario rows.
    assert scenario_df["scenario"].is_unique


def test_compute_scenario_metrics_safety_is_100_percent_in_every_scenario(df, plan, policy):
    scenario_df = compute_scenario_metrics(df, plan, policy)
    assert (scenario_df["boundary_coverage_percent"] == 100.0).all()
    assert (scenario_df["high_risk_preservation_percent"] == 100.0).all()


def test_compute_scenario_metrics_workflow_counts_sum_to_total(df, plan, policy):
    scenario_df = compute_scenario_metrics(df, plan, policy)
    assert int(scenario_df["workflow_count"].sum()) == df["workflow_id"].nunique()
    assert int(scenario_df["original_guardrail_calls"].sum()) == len(df)


def test_compute_scenario_metrics_returns_one_row_per_scenario_with_required_columns(df, plan, policy):
    scenario_df = compute_scenario_metrics(df, plan, policy)
    expected_scenarios = set(df["workflow_template"].unique())
    assert set(scenario_df["scenario"]) == expected_scenarios
    for col in [
        "scenario",
        "workflow_count",
        "original_guardrail_calls",
        "optimized_guardrail_calls",
        "call_reduction_percent",
        "latency_saved_percent",
        "cost_saved_percent",
        "boundary_coverage_percent",
        "high_risk_preservation_percent",
    ]:
        assert col in scenario_df.columns


def test_compute_scenario_metrics_safety_columns_are_always_100_percent(df, plan, policy):
    """Boundary coverage and high-risk preservation must hold at 100% in
    *every* individual scenario, not merely in the aggregate -- a template
    that happened to optimize aggressively must never do so at the expense
    of its own protected boundaries or high-risk checks."""
    scenario_df = compute_scenario_metrics(df, plan, policy)
    assert len(scenario_df) > 0
    assert (scenario_df["boundary_coverage_percent"] == 100.0).all()
    assert (scenario_df["high_risk_preservation_percent"] == 100.0).all()


def test_compute_scenario_metrics_call_counts_reconcile_with_overall(df, plan, policy):
    scenario_df = compute_scenario_metrics(df, plan, policy)
    assert int(scenario_df["original_guardrail_calls"].sum()) == len(df)
    assert int(scenario_df["workflow_count"].sum()) == df["workflow_id"].nunique()
