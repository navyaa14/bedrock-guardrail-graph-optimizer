"""
test_scenario_library.py
-------------------------
Tests proving each named scenario archetype (Part 2.4) is independently
runnable and independently testable, and behaves the way its name promises.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from synthetic_workflows import generate_scenario_workflows, SCENARIOS
from graph_builder import build_all_workflow_graphs, PROTECTED_BOUNDARIES
from graph_optimizer import load_policy, generate_optimized_plan

POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "guardrail_policy.yaml")


def test_all_six_scenarios_are_registered():
    expected = {
        "rag_customer_support",
        "financial_advisory",
        "code_generation",
        "healthcare_triage",
        "long_chain",
        "malformed_workflow",
    }
    assert expected == set(SCENARIOS.keys())
    for spec in SCENARIOS.values():
        assert "template" in spec and "behavior" in spec and "description" in spec


@pytest.mark.parametrize("scenario_name", sorted(SCENARIOS.keys()))
def test_scenario_generates_workflows_and_plan(scenario_name):
    """Every scenario must be independently runnable end-to-end."""
    df = generate_scenario_workflows(scenario_name, n_workflows=10, seed=5)
    assert df["workflow_id"].nunique() == 10
    graphs = build_all_workflow_graphs(df)
    policy = load_policy(POLICY_PATH)
    plan_df = generate_optimized_plan(graphs, policy)
    assert len(plan_df) > 0
    assert plan_df["recommended_decision"].notna().all()


def test_healthcare_triage_forces_high_risk_everywhere():
    df = generate_scenario_workflows("healthcare_triage", n_workflows=15, seed=5)
    assert (df["risk_level"] == "high").all()


def test_code_generation_has_no_internal_agent_hops():
    """code_generation's template intentionally has zero AGENT_TO_AGENT_INTERNAL
    edges -- it exists to prove that 0% call reduction is sometimes the
    correct, safe outcome rather than a bug."""
    df = generate_scenario_workflows("code_generation", n_workflows=10, seed=5)
    assert not (df["trust_boundary"] == "AGENT_TO_AGENT_INTERNAL").any()
    graphs = build_all_workflow_graphs(df)
    policy = load_policy(POLICY_PATH)
    plan_df = generate_optimized_plan(graphs, policy)
    optimized = plan_df["recommended_decision"].isin(["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"])
    assert not optimized.any()


def test_long_chain_has_at_least_six_hops():
    df = generate_scenario_workflows("long_chain", n_workflows=5, seed=5)
    graphs = build_all_workflow_graphs(df)
    for wg in graphs.values():
        assert len(wg.edges) >= 6


def test_malformed_workflow_combines_missing_boundary_and_orphan_branch_and_drift():
    """The deliberately malformed scenario must exhibit all three
    compounded failure modes at once: no AGENT_TO_FINAL_RESPONSE edge, an
    orphan (dead-end) branch, and a policy_id change mid-path."""
    df = generate_scenario_workflows("malformed_workflow", n_workflows=10, seed=5)
    graphs = build_all_workflow_graphs(df)

    found_missing_final = False
    found_orphan = False
    found_drift = False

    for (wf_id, req_id), wg in graphs.items():
        wf_rows = df[(df["workflow_id"] == wf_id) & (df["request_id"] == req_id)]
        if not (wf_rows["trust_boundary"] == "AGENT_TO_FINAL_RESPONSE").any():
            found_missing_final = True
        if len({p for p in wf_rows["policy_id"]}) > 1:
            found_drift = True
        types = {t for _, t in wg.all_paths_with_types()}
        if "orphan_branch" in types:
            found_orphan = True

    assert found_missing_final, "Expected malformed_workflow instances with no final-boundary edge."
    assert found_orphan, "Expected malformed_workflow instances with an orphan branch."
    assert found_drift, "Expected malformed_workflow instances with mid-path policy drift."


def test_malformed_workflow_never_silently_drops_checks_despite_compounded_failures():
    """Even with missing final boundary + orphan branch + policy drift all
    at once, every real guardrail-check row must still appear in the plan."""
    df = generate_scenario_workflows("malformed_workflow", n_workflows=15, seed=9)
    graphs = build_all_workflow_graphs(df)
    policy = load_policy(POLICY_PATH)
    plan_df = generate_optimized_plan(graphs, policy)

    raw_keys = set(zip(df["workflow_id"], df["request_id"], df["step_id"]))
    real_plan = plan_df[plan_df["step_id"] != -1]
    real_keys = set(zip(real_plan["workflow_id"], real_plan["request_id"], real_plan["step_id"]))
    assert raw_keys - real_keys == set(), "Some checks were silently dropped."


def test_financial_advisory_has_parallel_tool_fanout():
    df = generate_scenario_workflows("financial_advisory", n_workflows=5, seed=5)
    graphs = build_all_workflow_graphs(df)
    for wg in graphs.values():
        planner_successors = list(wg.graph.successors("PlannerAgent"))
        assert len(planner_successors) >= 2
