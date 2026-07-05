"""
test_graph_optimizer.py
------------------------
Tests proving the graph-based optimizer enforces every safety rule from
config/guardrail_policy.yaml: protected boundaries are always kept, high
risk is never silently skipped, drift/mutation/unsafe-prior-results always
block reuse, conflicting results escalate to human review, and every
reuse/skip carries an audit trail.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from synthetic_workflows import generate_synthetic_workflows
from graph_builder import build_all_workflow_graphs
from graph_optimizer import load_policy, generate_optimized_plan, detect_redundant_paths

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "guardrail_policy.yaml")


@pytest.fixture(scope="module")
def policy():
    return load_policy(CONFIG_PATH)


@pytest.fixture(scope="module")
def df():
    return generate_synthetic_workflows(n_workflows=150, seed=5)


@pytest.fixture(scope="module")
def graphs(df):
    return build_all_workflow_graphs(df)


@pytest.fixture(scope="module")
def plan(graphs, policy):
    return generate_optimized_plan(graphs, policy, strict_mode=False)


@pytest.fixture(scope="module")
def strict_plan(graphs, policy):
    return generate_optimized_plan(graphs, policy, strict_mode=True)


def test_keeps_user_to_agent_check(plan):
    rows = plan[plan["trust_boundary"] == "USER_TO_AGENT"]
    assert len(rows) > 0
    assert (rows["recommended_decision"] == "KEEP").all()


def test_keeps_agent_to_tool_check(plan):
    rows = plan[plan["trust_boundary"] == "AGENT_TO_TOOL"]
    assert len(rows) > 0
    assert (rows["recommended_decision"] == "KEEP").all()


def test_keeps_tool_to_agent_check(plan):
    rows = plan[plan["trust_boundary"] == "TOOL_TO_AGENT"]
    assert len(rows) > 0
    assert (rows["recommended_decision"] == "KEEP").all()


def test_keeps_agent_to_final_response_check(plan):
    rows = plan[plan["trust_boundary"] == "AGENT_TO_FINAL_RESPONSE"]
    real_rows = rows[rows["step_id"] != -1]
    assert len(real_rows) > 0
    assert (real_rows["recommended_decision"] == "KEEP").all()


def test_skips_low_risk_internal_duplicate_with_audit(plan):
    rows = plan[plan["recommended_decision"] == "SKIP_REDUNDANT_WITH_AUDIT"]
    assert len(rows) > 0
    assert (rows["risk_level"] == "low").all()
    assert (rows["audit_required"] == True).all()  # noqa: E712
    assert rows["reused_from_step_id"].notna().all()


def test_reuses_medium_risk_internal_duplicate(plan):
    rows = plan[plan["recommended_decision"] == "REUSE_PREVIOUS_DECISION"]
    assert len(rows) > 0
    assert set(rows["risk_level"].unique()).issubset({"low", "medium"})


def test_does_not_skip_high_risk_duplicate(plan):
    high_risk_dup = plan[
        (plan["risk_level"] == "high") & (plan["reused_from_step_id"].notna())
    ]
    assert len(high_risk_dup) > 0
    assert (high_risk_dup["recommended_decision"] != "SKIP_REDUNDANT_WITH_AUDIT").all()
    assert (high_risk_dup["recommended_decision"] != "REUSE_PREVIOUS_DECISION").all()


def test_does_not_reuse_if_text_changed(plan):
    rows = plan[plan["reason"] == "text_changed_since_previous_check"]
    assert len(rows) > 0
    assert (rows["recommended_decision"] == "KEEP").all()


def test_does_not_reuse_if_policy_changed(plan):
    rows = plan[plan["reason"] == "policy_drift_since_previous_check"]
    assert len(rows) > 0
    assert (rows["recommended_decision"] == "KEEP").all()


def test_does_not_reuse_if_previous_result_warn(plan):
    rows = plan[plan["reason"] == "previous_result_WARN_not_reusable"]
    if len(rows) > 0:
        assert (rows["recommended_decision"] == "KEEP").all()


def test_does_not_reuse_if_previous_result_block(plan):
    rows = plan[plan["reason"] == "previous_result_BLOCK_not_reusable"]
    if len(rows) > 0:
        assert (rows["recommended_decision"] == "KEEP").all()


def test_marks_conflicting_results_as_human_review(plan):
    rows = plan[plan["reason"] == "conflicting_guardrail_results"]
    assert len(rows) > 0
    assert (rows["recommended_decision"] == "HUMAN_REVIEW_REQUIRED").all()


def test_adds_move_to_boundary_when_final_check_missing(plan, df):
    missing_wf = df[df["behavior_pattern"] == "missing_final_boundary"]["workflow_id"].unique()
    move_rows = plan[(plan["recommended_decision"] == "MOVE_TO_BOUNDARY") & (plan["workflow_id"].isin(missing_wf))]
    assert len(move_rows) > 0
    assert (move_rows["trust_boundary"] == "AGENT_TO_FINAL_RESPONSE").all()


def test_adds_reused_from_step_id_for_every_reuse_or_skip(plan):
    reuse_or_skip = plan[plan["recommended_decision"].isin(["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"])]
    assert len(reuse_or_skip) > 0
    assert reuse_or_skip["reused_from_step_id"].notna().all()


def test_sets_audit_required_true_for_every_reuse_or_skip(plan):
    reuse_or_skip = plan[plan["recommended_decision"].isin(["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"])]
    assert len(reuse_or_skip) > 0
    assert (reuse_or_skip["audit_required"] == True).all()  # noqa: E712


def test_strict_mode_prevents_skip_redundant_with_audit(strict_plan):
    assert (strict_plan["recommended_decision"] != "SKIP_REDUNDANT_WITH_AUDIT").all()


def test_redundant_path_detection_is_path_aware(graphs, policy):
    redundant_df = detect_redundant_paths(graphs, policy)
    assert len(redundant_df) > 0
    for col in ["path_id", "path_nodes", "duplicate_reason", "recommended_decision"]:
        assert col in redundant_df.columns


# ---------------------------------------------------------------------------
# Full-plan-wide safety invariants (Part 3 hardening): scan the *entire*
# generated plan, not just a handful of matching rows, so a regression that
# only affects a subset of cases can't hide behind a partial assertion.
# ---------------------------------------------------------------------------


def test_protected_boundaries_never_have_skip_or_reuse_decision(plan):
    """No row on a protected trust boundary may ever be recommended for
    SKIP_REDUNDANT_WITH_AUDIT or REUSE_PREVIOUS_DECISION, anywhere in the
    plan -- protected boundaries are always KEEP or escalated to human
    review, never silently optimized away."""
    protected_rows = plan[plan["protected_boundary"] == True]  # noqa: E712
    assert len(protected_rows) > 0
    unsafe = protected_rows[
        protected_rows["recommended_decision"].isin(["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"])
    ]
    assert len(unsafe) == 0, f"Protected-boundary rows were optimized away: {unsafe.to_dict('records')[:3]}"


def test_move_to_boundary_rows_have_step_id_negative_one(plan):
    """MOVE_TO_BOUNDARY rows are synthetic recommendation rows (there was no
    real execution to recommend a placement change for) and must always be
    tagged step_id == -1 so plan-integrity accounting can exclude them from
    the real-execution comparison (see test_plan_integrity.py)."""
    move_rows = plan[plan["recommended_decision"] == "MOVE_TO_BOUNDARY"]
    assert len(move_rows) > 0
    assert (move_rows["step_id"] == -1).all()


def test_move_to_boundary_rows_are_protected_and_audit_required(plan):
    move_rows = plan[plan["recommended_decision"] == "MOVE_TO_BOUNDARY"]
    assert len(move_rows) > 0
    assert (move_rows["protected_boundary"] == True).all()  # noqa: E712
    assert (move_rows["audit_required"] == True).all()  # noqa: E712
    assert (move_rows["trust_boundary"] == "AGENT_TO_FINAL_RESPONSE").all()
    assert (move_rows["safe_to_optimize"] == False).all()  # noqa: E712


def test_warn_or_block_previous_result_never_reused_across_full_plan(plan):
    """Scan every row whose reason cites an unsafe (WARN/BLOCK) previous
    result and confirm none of them were reused or skipped -- not just the
    handful checked by test_does_not_reuse_if_previous_result_warn/block."""
    unsafe_prior_rows = plan[plan["reason"].str.startswith("previous_result_", na=False)]
    assert len(unsafe_prior_rows) > 0
    assert (unsafe_prior_rows["recommended_decision"] == "KEEP").all()
    assert not unsafe_prior_rows["recommended_decision"].isin(
        ["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"]
    ).any()


def test_policy_drift_never_reused_across_full_plan(plan):
    drift_rows = plan[plan["reason"] == "policy_drift_since_previous_check"]
    assert len(drift_rows) > 0
    assert not drift_rows["recommended_decision"].isin(
        ["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"]
    ).any()


def test_text_changed_never_reused_across_full_plan(plan):
    mutation_rows = plan[plan["reason"] == "text_changed_since_previous_check"]
    assert len(mutation_rows) > 0
    assert not mutation_rows["recommended_decision"].isin(
        ["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"]
    ).any()


def test_every_skip_or_reuse_row_has_reused_from_step_id_and_audit_required_at_scale(graphs, policy):
    """Same invariant as test_adds_reused_from_step_id_for_every_reuse_or_skip
    / test_sets_audit_required_true_for_every_reuse_or_skip, but re-derived
    from a larger, independently-generated corpus (300 workflows, a
    different seed) so the invariant is proven to hold generally, not just
    for the module-scoped 150-workflow fixture already used elsewhere in
    this file."""
    df = generate_synthetic_workflows(n_workflows=300, seed=77)
    big_graphs = build_all_workflow_graphs(df)
    big_plan = generate_optimized_plan(big_graphs, policy, strict_mode=False)

    reuse_or_skip = big_plan[
        big_plan["recommended_decision"].isin(["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"])
    ]
    assert len(reuse_or_skip) > 0
    assert reuse_or_skip["reused_from_step_id"].notna().all()
    assert (reuse_or_skip["audit_required"] == True).all()  # noqa: E712

