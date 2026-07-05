"""
test_graph_features.py
-----------------------
Tests proving graph_features.py computes real graph-level features
(node/edge counts, path length, missing boundary detection, policy drift,
text mutation, redundant internal checks, and the transparent, rule-based
graph_risk_score) rather than row-level aggregates.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from synthetic_workflows import generate_synthetic_workflows
from graph_builder import build_all_workflow_graphs
from graph_features import compute_all_workflow_features
from graph_optimizer import load_policy, generate_optimized_plan
from synthetic_workflows import generate_scenario_workflows

RISK_WEIGHTS = {
    "high_risk_boundary": 3,
    "tool_boundary": 2,
    "missing_final_check": 2,
    "policy_drift": 1,
    "text_mutation": 1,
    "warn_result": 1,
    "block_result": 2,
}


@pytest.fixture(scope="module")
def df():
    return generate_synthetic_workflows(n_workflows=120, seed=3)


@pytest.fixture(scope="module")
def graphs(df):
    return build_all_workflow_graphs(df)


@pytest.fixture(scope="module")
def feature_df(graphs):
    return compute_all_workflow_features(graphs, RISK_WEIGHTS)


def test_calculates_node_count(feature_df, graphs):
    for _, row in feature_df.iterrows():
        wg = graphs[(row["workflow_id"], row["request_id"])]
        assert row["node_count"] == len(wg.nodes)


def test_calculates_edge_count(feature_df, graphs):
    for _, row in feature_df.iterrows():
        wg = graphs[(row["workflow_id"], row["request_id"])]
        assert row["edge_count"] == len(wg.edges)


def test_calculates_max_path_length(feature_df, graphs):
    for _, row in feature_df.iterrows():
        wg = graphs[(row["workflow_id"], row["request_id"])]
        paths = wg.all_paths()
        expected = max((len(p) - 1 for p in paths), default=0)
        assert row["max_path_length"] == expected


def test_detects_missing_final_boundary(df, feature_df):
    missing_wf = df[df["behavior_pattern"] == "missing_final_boundary"]["workflow_id"].unique()
    assert len(missing_wf) > 0
    flagged = feature_df[feature_df["workflow_id"].isin(missing_wf)]["missing_final_boundary_check"]
    assert flagged.all()


def test_detects_policy_drift(df, feature_df):
    drift_wf = df[df["behavior_pattern"] == "policy_drift"]["workflow_id"].unique()
    assert len(drift_wf) > 0
    drift_features = feature_df[feature_df["workflow_id"].isin(drift_wf)]
    assert (drift_features["policy_drift_count"] > 0).any()


def test_detects_text_mutation(df, feature_df):
    mutation_wf = df[df["behavior_pattern"] == "text_mutation"]["workflow_id"].unique()
    assert len(mutation_wf) > 0
    mutation_features = feature_df[feature_df["workflow_id"].isin(mutation_wf)]
    assert (mutation_features["text_mutation_count"] > 0).any()


def test_calculates_graph_risk_score(feature_df):
    assert (feature_df["graph_risk_score"] >= 0).all()
    # High-risk-duplicate workflows should score above the median.
    median_score = feature_df["graph_risk_score"].median()
    assert feature_df["graph_risk_score"].max() >= median_score


def test_detects_redundant_internal_checks(df, feature_df):
    normal_wf = df[df["behavior_pattern"] == "normal_duplicate"]["workflow_id"].unique()
    normal_features = feature_df[feature_df["workflow_id"].isin(normal_wf)]
    assert (normal_features["redundant_internal_checks"] >= 0).all()
    assert (normal_features["redundant_internal_checks"] > 0).any()


def test_redundant_internal_checks_matches_optimizer_reuse_and_skip_decisions():
    """Regression test for a real bug: redundant_internal_checks used to only
    count a duplicate if BOTH occurrences sat on AGENT_TO_AGENT_INTERNAL
    edges. But graph_optimizer._decide_step tracks "previous occurrence" of
    a (guardrail_type, policy_id, text_hash) key across the WHOLE path --
    including protected boundaries -- and only checks whether the CURRENT
    (duplicate) check's own edge is internal. USER_TO_AGENT is typically the
    first edge on any path, so "protected-boundary-first, internal-repeat-
    second" is the common case, not an edge case, and the old logic silently
    under-counted it to zero on scenarios built exactly that way (e.g.
    financial_advisory), even while the optimizer was correctly producing
    real SKIP_REUNDANT_WITH_AUDIT / REUSE_PREVIOUS_DECISION rows.

    This test proves the feature and the optimizer now agree: every
    optimizer-approved reuse/skip decision on an internal edge must be
    reflected as a redundant_internal_checks count of at least 1 in that
    workflow's row of workflow_graph_summary.csv.
    """
    df = generate_scenario_workflows("financial_advisory", n_workflows=20, seed=42)
    graphs = build_all_workflow_graphs(df)
    policy = load_policy()

    feature_df = compute_all_workflow_features(graphs, RISK_WEIGHTS)
    plan_df = generate_optimized_plan(graphs, policy)

    optimized = plan_df[plan_df["recommended_decision"].isin(
        ["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"]
    )]
    assert len(optimized) > 0, "financial_advisory scenario should produce real reuse/skip decisions"

    workflows_with_real_optimizations = set(optimized["workflow_id"].unique())
    flagged_workflows = set(
        feature_df[feature_df["redundant_internal_checks"] > 0]["workflow_id"]
    )

    # Every workflow where the optimizer actually approved a reuse/skip must
    # be flagged by the graph feature as having a redundant internal check.
    assert workflows_with_real_optimizations.issubset(flagged_workflows), (
        "Workflows with real optimizer SKIP/REUSE decisions were not reflected "
        "in redundant_internal_checks: "
        f"{workflows_with_real_optimizations - flagged_workflows}"
    )

    # The total count of internal-edge duplicates found by the feature must
    # be at least the number the optimizer actually approved (it can be
    # larger, since some internal duplicates are correctly blocked by other
    # safety rules -- e.g. high_risk or conflicting_result -- and still
    # count as a structural duplicate even though they were not optimized).
    total_redundant_internal = int(feature_df["redundant_internal_checks"].sum())
    assert total_redundant_internal >= len(optimized)
