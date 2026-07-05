"""
explanations.py
----------------
Generates plain-language, audit-friendly explanations for the highest-impact
workflows in the optimized guardrail plan. Each explanation is grounded in
the graph: which path was involved, which trust boundaries were preserved,
and why a duplicate was judged safe or unsafe to optimize.

These explanations are meant to support human review and audit trails, not
to replace them -- every reuse/skip decision remains fully traceable back to
a specific prior step_id.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from graph_builder import WorkflowGraph


def _path_summary(wg: WorkflowGraph, graph_path_id: str) -> str:
    if wg is None:
        return ""
    # graph_optimizer.py tags path ids as "{request_id}_p{idx}" for real
    # final paths and "{request_id}_o{idx}" for dead-end orphan branches,
    # both sharing one index counter over all_paths_with_types(). Mirror
    # that scheme exactly so lookups agree.
    for path_idx, (path, path_type) in enumerate(wg.all_paths_with_types()):
        prefix = "p" if path_type == "final" else "o"
        if f"{wg.request_id}_{prefix}{path_idx}" == graph_path_id:
            return " -> ".join(path)
    return ""


def _build_explanation_text(
    workflow_id: str,
    path_summary: str,
    guardrail_type: str,
    policy_id: str,
    duplicate_count: int,
    decision: str,
    boundaries_preserved: List[str],
) -> str:
    boundary_txt = ", ".join(boundaries_preserved) if boundaries_preserved else "no additional boundaries"

    if decision == "SKIP_REDUNDANT_WITH_AUDIT":
        action_txt = (
            f"the optimizer recommends SKIP_REDUNDANT_WITH_AUDIT for {duplicate_count} downstream "
            f"check(s) on internal handoffs"
        )
        safety_txt = (
            "the duplicate checks occurred only on AGENT_TO_AGENT_INTERNAL edges, the previous result "
            "was PASS, the text hash did not change, and risk was low"
        )
    elif decision == "REUSE_PREVIOUS_DECISION":
        action_txt = f"the optimizer recommends REUSE_PREVIOUS_DECISION for {duplicate_count} downstream check(s)"
        safety_txt = "risk was medium, so the prior decision is reused with an audit trail rather than fully skipped"
    elif decision == "HUMAN_REVIEW_REQUIRED":
        action_txt = "the optimizer flags this path for HUMAN_REVIEW_REQUIRED"
        safety_txt = "the duplicate involved high risk or a conflicting guardrail result, so it is never auto-optimized"
    else:
        action_txt = f"the optimizer recommends {decision}"
        safety_txt = "the check falls on a protected trust boundary and is always preserved"

    return (
        f"Workflow {workflow_id} contains a path {path_summary}. The same {guardrail_type} under "
        f"policy {policy_id} was executed multiple times across internal agent handoffs. Because "
        f"{safety_txt}, {action_txt}. Checks at {boundary_txt} are preserved."
    )


def generate_explanations(
    plan_df: pd.DataFrame,
    graphs: Dict[Tuple[str, str], WorkflowGraph],
    top_n: int = 25,
) -> List[dict]:
    """Generate explanation records for the top-N highest-impact workflows."""
    real_rows = plan_df[plan_df["step_id"] != -1].copy()

    impact = (
        real_rows.assign(
            is_opt=real_rows["recommended_decision"].isin(["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"])
        )
        .groupby("workflow_id")["is_opt"]
        .sum()
        .sort_values(ascending=False)
    )
    top_workflows = list(impact.head(top_n).index)

    explanations = []
    for workflow_id in top_workflows:
        wf_rows = real_rows[real_rows["workflow_id"] == workflow_id]
        if wf_rows.empty:
            continue

        wf_request_id = wf_rows.iloc[0]["request_id"]
        wg = graphs.get((workflow_id, wf_request_id))
        if wg is None:
            continue

        dup_rows = wf_rows[wf_rows["reused_from_step_id"].notna()]
        if dup_rows.empty:
            continue

        top_group = (
            dup_rows.groupby(["guardrail_type", "policy_id", "graph_path_id"]).size().sort_values(ascending=False)
        )
        (guardrail_type, policy_id, graph_path_id) = top_group.index[0]
        group_rows = dup_rows[
            (dup_rows["guardrail_type"] == guardrail_type)
            & (dup_rows["policy_id"] == policy_id)
            & (dup_rows["graph_path_id"] == graph_path_id)
        ]

        decision = group_rows.iloc[-1]["recommended_decision"]
        duplicate_count = len(group_rows)

        boundaries_preserved = sorted(
            wf_rows[wf_rows["recommended_decision"] == "KEEP"]["trust_boundary"].unique().tolist()
        )

        path_summary = _path_summary(wg, graph_path_id)

        non_keep_rows = group_rows[group_rows["recommended_decision"] != "KEEP"]
        multiplier = 1.0 if decision == "SKIP_REDUNDANT_WITH_AUDIT" else 0.8
        latency_saved_ms = round(float(non_keep_rows["original_latency_ms"].sum()) * multiplier, 2)
        cost_saved_usd = round(float(non_keep_rows["original_cost_usd"].sum()) * multiplier, 6)

        explanation_text = _build_explanation_text(
            workflow_id, path_summary, guardrail_type, policy_id, duplicate_count, decision, boundaries_preserved
        )

        explanations.append(
            {
                "workflow_id": workflow_id,
                "path_summary": path_summary,
                "duplicate_guardrail_pattern": f"{guardrail_type} / {policy_id} repeated {duplicate_count}x",
                "recommended_action": decision,
                "safety_reasoning": explanation_text,
                "boundary_checks_preserved": boundaries_preserved,
                "latency_saved_ms": latency_saved_ms,
                "cost_saved_usd": cost_saved_usd,
                "audit_required": bool(group_rows["audit_required"].any()),
                "human_review_required": bool((wf_rows["recommended_decision"] == "HUMAN_REVIEW_REQUIRED").any()),
            }
        )

    return explanations
