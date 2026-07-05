"""
test_plan_integrity.py
-----------------------
The single most important test in this repo.

Runs the full pipeline end-to-end at several (workflow count, seed)
combinations and asserts the optimized guardrail plan is a PERFECT
one-to-one accounting of the raw synthetic trace: every real
(workflow_id, request_id, step_id) that was executed appears in the
optimized plan exactly once -- zero missing, zero duplicate, zero extra.

This is the regression test for the Part 1 correctness bug: guardrail
checks on a dead-end orphan branch (one that never reaches FinalAnswer)
used to be silently dropped from the plan whenever a *different* branch of
the same workflow *did* reach FinalAnswer, because `all_paths()` only
enumerated root->final paths and its fallback to leaf nodes only ever
triggered when NO final node existed anywhere in the graph.

`step_id == -1` rows are the MOVE_TO_BOUNDARY recommendation rows added by
graph_optimizer.py when a workflow has no guardrail check at all on its
AGENT_TO_FINAL_RESPONSE boundary. They are not real executions and are
excluded from this comparison by design (see CANONICAL BASE note in the
project brief).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from synthetic_workflows import generate_synthetic_workflows
from graph_builder import build_all_workflow_graphs
from graph_optimizer import load_policy, generate_optimized_plan
from graph_metrics import compute_scenario_metrics

POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "guardrail_policy.yaml")


def _assert_plan_is_exact_accounting(n_workflows: int, seed: int) -> None:
    policy = load_policy(POLICY_PATH)
    df = generate_synthetic_workflows(n_workflows=n_workflows, seed=seed)
    graphs = build_all_workflow_graphs(df)
    plan_df = generate_optimized_plan(graphs, policy)

    raw_keys = set(zip(df["workflow_id"], df["request_id"], df["step_id"]))

    real_plan_rows = plan_df[plan_df["step_id"] != -1]
    real_plan_keys = set(
        zip(real_plan_rows["workflow_id"], real_plan_rows["request_id"], real_plan_rows["step_id"])
    )

    missing = raw_keys - real_plan_keys
    extra = real_plan_keys - raw_keys

    # Duplicate detection: real_plan_keys is a set, so instead check the
    # plan's real rows have no duplicate (workflow_id, request_id, step_id)
    # tuples at the row level.
    dup_count = len(real_plan_rows) - len(real_plan_keys)

    assert not missing, (
        f"[n_workflows={n_workflows}, seed={seed}] {len(missing)} step(s) present in the raw trace "
        f"but MISSING from the optimized plan -- guardrail checks are being silently dropped. "
        f"Examples: {list(missing)[:5]}"
    )
    assert not extra, (
        f"[n_workflows={n_workflows}, seed={seed}] {len(extra)} step(s) in the optimized plan do not "
        f"correspond to any real raw trace row. Examples: {list(extra)[:5]}"
    )
    assert dup_count == 0, (
        f"[n_workflows={n_workflows}, seed={seed}] {dup_count} duplicate (workflow_id, request_id, "
        f"step_id) row(s) found in the optimized plan."
    )


@pytest.mark.parametrize(
    "n_workflows,seed",
    [
        (500, 42),
        (2000, 7),
        (2000, 99),
        (1000, 123),
        (50, 5),
    ],
)
def test_plan_is_exact_accounting_of_raw_trace(n_workflows, seed):
    """Zero missing, zero duplicate, zero extra plan rows, across multiple
    random topologies -- not just one seed. This must fail loudly if the
    edge-coverage guarantee in all_paths_with_types() ever regresses."""
    _assert_plan_is_exact_accounting(n_workflows, seed)


def test_move_to_boundary_rows_are_extra_and_excluded_from_raw_accounting():
    """MOVE_TO_BOUNDARY rows (step_id == -1) are recommendation rows added
    by the optimizer, not real executions -- they must never appear in the
    raw trace, and every one of them must be tagged step_id == -1 so the
    exact-accounting check above can safely exclude them."""
    policy = load_policy(POLICY_PATH)
    df = generate_synthetic_workflows(n_workflows=300, seed=3)
    graphs = build_all_workflow_graphs(df)
    plan_df = generate_optimized_plan(graphs, policy)

    move_rows = plan_df[plan_df["recommended_decision"] == "MOVE_TO_BOUNDARY"]
    assert len(move_rows) > 0
    assert (move_rows["step_id"] == -1).all()

    # None of the raw trace's real step_ids are ever -1, so MOVE_TO_BOUNDARY
    # rows can never collide with a real (workflow_id, request_id, step_id).
    assert not (df["step_id"] == -1).any()

    real_plan_rows = plan_df[plan_df["step_id"] != -1]
    assert len(real_plan_rows) + len(move_rows) == len(plan_df)


def test_scenario_metrics_reconciles_with_overall_totals():
    """outputs/scenario_metrics.csv (Part 5 hardening) must be a true
    breakdown of the overall run: original/optimized call counts summed
    across every scenario must equal the whole-corpus totals exactly, and
    every scenario must independently show 100% boundary coverage and
    100% high-risk preservation."""
    policy = load_policy(POLICY_PATH)
    df = generate_synthetic_workflows(n_workflows=300, seed=42)
    graphs = build_all_workflow_graphs(df)
    plan_df = generate_optimized_plan(graphs, policy)

    scenario_df = compute_scenario_metrics(df, plan_df, policy)
    assert len(scenario_df) > 0
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

    assert int(scenario_df["original_guardrail_calls"].sum()) == len(df)
    assert int(scenario_df["workflow_count"].sum()) == df["workflow_id"].nunique()
    assert (scenario_df["boundary_coverage_percent"] == 100.0).all()
    assert (scenario_df["high_risk_preservation_percent"] == 100.0).all()


def test_plan_never_drops_orphan_branch_checks():
    """Targeted regression test: a workflow with BOTH a real final path and
    a dead-end orphan branch must have every check on the dead branch show
    up in the optimized plan with a decision (never silently vanish)."""
    policy = load_policy(POLICY_PATH)
    df = generate_synthetic_workflows(n_workflows=80, seed=11)
    graphs = build_all_workflow_graphs(df)
    plan_df = generate_optimized_plan(graphs, policy)

    found_mixed = False
    for wg in graphs.values():
        paths_with_types = wg.all_paths_with_types()
        types = {t for _, t in paths_with_types}
        if "orphan_branch" in types and "final" in types:
            found_mixed = True
            orphan_edges = set()
            for p, t in paths_with_types:
                if t == "orphan_branch":
                    for i in range(len(p) - 1):
                        orphan_edges.add((p[i], p[i + 1]))
            for edge_key in orphan_edges:
                edge = wg.edges[edge_key]
                for check in edge.guardrail_checks:
                    step_id = check.get("step_id")
                    match = plan_df[
                        (plan_df["workflow_id"] == wg.workflow_id)
                        & (plan_df["request_id"] == wg.request_id)
                        & (plan_df["step_id"] == step_id)
                    ]
                    assert len(match) == 1, (
                        f"Orphan-branch check step_id={step_id} for "
                        f"{wg.workflow_id}/{wg.request_id} missing from optimized plan."
                    )
                    assert match.iloc[0]["recommended_decision"] is not None

    assert found_mixed, "Expected at least one mixed (final + orphan_branch) workflow in the generated corpus."
