"""
graph_optimizer.py
-------------------
Graph-first guardrail placement optimizer.

Two responsibilities, both operating on the directed WorkflowGraph objects
from graph_builder.py (NOT flat row comparison):

1. detect_redundant_paths(): enumerates every UserInput -> FinalAnswer path
   in each workflow graph and walks it in execution order, looking for a
   guardrail_type that repeats along that same path. This is graph-path-
   aware: two checks are only considered a "duplicate along a path" if they
   actually occur on the same path through the graph, not merely somewhere
   in the same workflow_id.

2. generate_optimized_plan(): produces the final, config-driven, per-step
   recommendation (KEEP / REUSE_PREVIOUS_DECISION / SKIP_REDUNDANT_WITH_AUDIT
   / MOVE_TO_BOUNDARY / HUMAN_REVIEW_REQUIRED) for every guardrail check in
   every workflow, guided by config/guardrail_policy.yaml.

Safe wording only: this module RECOMMENDS placements and always requires an
audit trail (reused_from_step_id + audit_required) for any reuse or skip.
It never claims to "remove" or "bypass" a guardrail, and it never claims a
security guarantee -- protected trust boundaries and high-risk paths are
always kept or escalated to human review.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from graph_builder import WorkflowGraph

DECISIONS = [
    "KEEP",
    "REUSE_PREVIOUS_DECISION",
    "SKIP_REDUNDANT_WITH_AUDIT",
    "MOVE_TO_BOUNDARY",
    "HUMAN_REVIEW_REQUIRED",
]


def load_policy(path: str = "config/guardrail_policy.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Path-level redundancy detection
# ---------------------------------------------------------------------------


def _path_ordered_checks(wg: WorkflowGraph, path: List[str]) -> List[Tuple]:
    """Return (edge, check_dict) pairs for every guardrail check along `path`,
    in execution (step_id) order."""
    items = []
    for i in range(len(path) - 1):
        edge = wg.edges.get((path[i], path[i + 1]))
        if edge is None:
            continue
        for check in edge.guardrail_checks:
            items.append((edge, check))
    items.sort(key=lambda t: t[1].get("step_id", 0))
    return items


def _classify_duplicate(
    check: dict, prev: dict, edge, protected: set, never_reuse: set, path_type: str = "final"
) -> Tuple[bool, List[str], str]:
    """Return (optimization_allowed, blocked_by_list, recommended_decision) for
    a duplicate `check` given the `prev` occurrence of the same guardrail_type
    earlier on the same path."""
    blocked_by: List[str] = []

    if edge.trust_boundary in protected:
        blocked_by.append("protected_boundary")
    if check.get("risk_level") == "high" or prev.get("risk_level") == "high" or edge.edge_risk_level == "high":
        blocked_by.append("high_risk")
    if bool(check.get("text_changed")) or check.get("text_hash") != prev.get("text_hash"):
        blocked_by.append("text_changed")
    if check.get("policy_id") != prev.get("policy_id"):
        blocked_by.append("policy_drift")
    if prev.get("guardrail_result") in never_reuse:
        blocked_by.append("previous_unsafe_result")
    if path_type == "orphan_branch":
        blocked_by.append("orphan_branch")

    conflicting_results = prev.get("guardrail_result") != check.get("guardrail_result")

    if "high_risk" in blocked_by:
        decision = "HUMAN_REVIEW_REQUIRED"
    elif conflicting_results:
        decision = "HUMAN_REVIEW_REQUIRED"
        blocked_by.append("conflicting_result")
    elif blocked_by:
        # Any blocking reason -- including a dead-end orphan branch, which
        # we can never safely optimize since we can't confirm it reaches a
        # protected final boundary -- defaults to KEEP, never a silent drop.
        decision = "KEEP"
    else:
        decision = "SKIP_REDUNDANT_WITH_AUDIT" if check.get("risk_level") == "low" else "REUSE_PREVIOUS_DECISION"

    optimization_allowed = decision in ("SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION")
    return optimization_allowed, blocked_by, decision


def detect_redundant_paths(graphs: Dict[Tuple[str, str], WorkflowGraph], policy: dict) -> pd.DataFrame:
    """Graph-path-aware redundant guardrail detection. Returns outputs/redundant_paths.csv rows."""
    protected = set(policy["protected_boundaries"])
    never_reuse = set(policy["never_reuse_previous_results"])

    rows = []
    for wg in graphs.values():
        paths_with_types = wg.all_paths_with_types()
        for path_idx, (path, path_type) in enumerate(paths_with_types):
            prefix = "p" if path_type == "final" else "o"
            path_id = f"{wg.request_id}_{prefix}{path_idx}"
            items = _path_ordered_checks(wg, path)
            seen: Dict[str, dict] = {}
            for edge, check in items:
                gtype = check.get("guardrail_type")
                prev = seen.get(gtype)
                if prev is not None:
                    optimization_allowed, blocked_by, decision = _classify_duplicate(
                        check, prev, edge, protected, never_reuse, path_type=path_type
                    )
                    reason = "same_guardrail_same_text_same_policy" if not blocked_by else ";".join(blocked_by)
                    rows.append(
                        {
                            "workflow_id": wg.workflow_id,
                            "request_id": wg.request_id,
                            "path_id": path_id,
                            "path_type": path_type,
                            "path_nodes": " -> ".join(path),
                            "guardrail_type": gtype,
                            "policy_id": check.get("policy_id"),
                            "text_hash": check.get("text_hash"),
                            "first_step_id": prev.get("step_id"),
                            "duplicate_step_id": check.get("step_id"),
                            "duplicate_reason": reason,
                            "optimization_allowed": optimization_allowed,
                            "blocked_by": ";".join(blocked_by),
                            "recommended_decision": decision,
                        }
                    )
                seen[gtype] = check

    return pd.DataFrame(
        rows,
        columns=[
            "workflow_id",
            "request_id",
            "path_id",
            "path_type",
            "path_nodes",
            "guardrail_type",
            "policy_id",
            "text_hash",
            "first_step_id",
            "duplicate_step_id",
            "duplicate_reason",
            "optimization_allowed",
            "blocked_by",
            "recommended_decision",
        ],
    )


# ---------------------------------------------------------------------------
# Per-step optimized plan
# ---------------------------------------------------------------------------


def _decide_step(
    check: dict,
    edge,
    prev: Optional[dict],
    protected: set,
    never_reuse: set,
    skip_allowed_risk: set,
    reuse_allowed_risk: set,
    strict_mode: bool,
    path_type: str = "final",
) -> Tuple[str, str, Optional[int], bool]:
    """Return (decision, reason, reused_from_step_id, audit_required) for one check."""

    # A. Protected boundary rule -- always kept regardless of duplication.
    if edge.trust_boundary in protected:
        return "KEEP", "protected_trust_boundary_always_kept", None, False

    # B. High-risk rule -- never silently skipped.
    is_high_risk = check.get("risk_level") == "high" or edge.edge_risk_level == "high"
    if is_high_risk:
        if prev is not None:
            return "HUMAN_REVIEW_REQUIRED", "high_risk_duplicate_requires_human_review", prev.get("step_id"), True
        return "KEEP", "high_risk_check_kept", None, False

    if prev is None:
        return "KEEP", "first_occurrence_on_this_graph_path", None, False

    # C. Previous unsafe result rule.
    if prev.get("guardrail_result") in never_reuse:
        return "KEEP", f"previous_result_{prev.get('guardrail_result')}_not_reusable", None, False

    # D. Text mutation rule.
    if bool(check.get("text_changed")) or check.get("text_hash") != prev.get("text_hash"):
        return "KEEP", "text_changed_since_previous_check", None, False

    # E. Policy drift rule.
    if check.get("policy_id") != prev.get("policy_id"):
        return "KEEP", "policy_drift_since_previous_check", None, False

    # I. Conflict rule.
    if prev.get("guardrail_result") != check.get("guardrail_result"):
        return "HUMAN_REVIEW_REQUIRED", "conflicting_guardrail_results", prev.get("step_id"), True

    # Safe duplicate: same guardrail_type + policy_id + text_hash, internal
    # boundary, previous result reusable, no drift/mutation, not high risk.
    risk = check.get("risk_level")

    if strict_mode:
        if risk in reuse_allowed_risk:
            return "REUSE_PREVIOUS_DECISION", "strict_mode_reuse_previous_decision", prev.get("step_id"), True
        return "HUMAN_REVIEW_REQUIRED", "strict_mode_escalates_uncertain_case", prev.get("step_id"), True

    if risk in skip_allowed_risk:
        return (
            "SKIP_REDUNDANT_WITH_AUDIT",
            "low_risk_internal_duplicate_same_text_same_policy_previous_pass",
            prev.get("step_id"),
            True,
        )
    if risk in reuse_allowed_risk:
        return (
            "REUSE_PREVIOUS_DECISION",
            "medium_risk_internal_duplicate_reuse_prior_decision",
            prev.get("step_id"),
            True,
        )
    return "HUMAN_REVIEW_REQUIRED", "unclassified_risk_escalated_for_safety", prev.get("step_id"), True


def generate_optimized_plan(
    graphs: Dict[Tuple[str, str], WorkflowGraph],
    policy: dict,
    strict_mode: bool = False,
) -> pd.DataFrame:
    """Produce the graph-aware optimized guardrail placement plan."""
    protected = set(policy["protected_boundaries"])
    never_reuse = set(policy["never_reuse_previous_results"])
    skip_allowed_risk = set(policy["skip_allowed_risk_levels"])
    reuse_allowed_risk = set(policy["reuse_allowed_risk_levels"])

    rows = []

    for wg in graphs.values():
        paths_with_types = wg.all_paths_with_types()

        # Walk each path independently so duplicate detection never crosses
        # unrelated branches (true graph-path awareness). Orphan branches
        # (dead-end hops that never reach FinalAnswer) are walked too, so
        # their checks are always routed through the decision engine and
        # never silently dropped -- see all_paths_with_types().
        emitted_step_ids = set()
        for path_idx, (path, path_type) in enumerate(paths_with_types):
            prefix = "p" if path_type == "final" else "o"
            path_id = f"{wg.request_id}_{prefix}{path_idx}"
            seen: Dict[str, dict] = {}
            items = _path_ordered_checks(wg, path)
            for edge, check in items:
                gtype = check.get("guardrail_type")
                prev = seen.get(gtype)
                decision, reason, reused_from, audit_required = _decide_step(
                    check,
                    edge,
                    prev,
                    protected,
                    never_reuse,
                    skip_allowed_risk,
                    reuse_allowed_risk,
                    strict_mode,
                    path_type=path_type,
                )
                # G. Orphan branch rule: the check still runs through the
                # full decision engine above (so HUMAN_REVIEW_REQUIRED
                # escalations from high-risk/conflicting-result rules are
                # preserved), but on a dead-end branch that never reaches a
                # final node we can never confirm downstream coverage, so
                # any would-be optimization is conservatively downgraded to
                # KEEP -- it must never silently vanish from the plan.
                if path_type == "orphan_branch" and decision in (
                    "SKIP_REDUNDANT_WITH_AUDIT",
                    "REUSE_PREVIOUS_DECISION",
                ):
                    decision = "KEEP"
                    reason = f"orphan_branch_dead_end_conservatively_kept;{reason}"
                    audit_required = False
                seen[gtype] = check

                # A check on a shared edge (e.g. a branch-convergence edge
                # traversed by more than one path) is one real execution --
                # emit it once, keyed globally, so metrics never double-count.
                step_key = (wg.workflow_id, wg.request_id, check.get("step_id"))
                if step_key in emitted_step_ids:
                    continue
                emitted_step_ids.add(step_key)

                rows.append(
                    {
                        "workflow_id": wg.workflow_id,
                        "request_id": wg.request_id,
                        "step_id": check.get("step_id"),
                        "node_id": check.get("node_id"),
                        "source_node": edge.source_node,
                        "target_node": edge.target_node,
                        "agent_name": check.get("agent_name"),
                        "guardrail_type": gtype,
                        "policy_id": check.get("policy_id"),
                        "text_hash": check.get("text_hash"),
                        "trust_boundary": edge.trust_boundary,
                        "risk_level": check.get("risk_level"),
                        "original_latency_ms": check.get("latency_ms"),
                        "original_cost_usd": check.get("estimated_cost_usd"),
                        "recommended_decision": decision,
                        "reason": reason,
                        "reused_from_step_id": reused_from,
                        "safe_to_optimize": decision in ("SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"),
                        "graph_path_id": path_id,
                        "path_type": path_type,
                        "protected_boundary": edge.trust_boundary in protected,
                        "audit_required": audit_required,
                    }
                )

        # H. Missing final boundary rule: recommend adding a check.
        if not wg.has_final_boundary_check():
            all_checks = []
            for e in wg.edges.values():
                all_checks.extend(e.guardrail_checks)
            fallback_gtype = all_checks[0]["guardrail_type"] if all_checks else "GROUNDING_CHECK"
            final_nodes = wg.final_nodes()
            final_node = final_nodes[0] if final_nodes else "FinalAnswer"
            rows.append(
                {
                    "workflow_id": wg.workflow_id,
                    "request_id": wg.request_id,
                    "step_id": -1,
                    "node_id": f"{wg.request_id}_{final_node}_boundary_gap",
                    "source_node": "?",
                    "target_node": final_node,
                    "agent_name": final_node,
                    "guardrail_type": fallback_gtype,
                    "policy_id": None,
                    "text_hash": None,
                    "trust_boundary": "AGENT_TO_FINAL_RESPONSE",
                    "risk_level": "unknown",
                    "original_latency_ms": 0.0,
                    "original_cost_usd": 0.0,
                    "recommended_decision": "MOVE_TO_BOUNDARY",
                    "reason": "no_guardrail_check_found_on_agent_to_final_response_boundary",
                    "reused_from_step_id": None,
                    "safe_to_optimize": False,
                    "graph_path_id": f"{wg.request_id}_p0",
                    "path_type": "boundary_gap",
                    "protected_boundary": True,
                    "audit_required": True,
                }
            )

    columns = [
        "workflow_id",
        "request_id",
        "step_id",
        "node_id",
        "source_node",
        "target_node",
        "agent_name",
        "guardrail_type",
        "policy_id",
        "text_hash",
        "trust_boundary",
        "risk_level",
        "original_latency_ms",
        "original_cost_usd",
        "recommended_decision",
        "reason",
        "reused_from_step_id",
        "safe_to_optimize",
        "graph_path_id",
        "path_type",
        "protected_boundary",
        "audit_required",
    ]
    return pd.DataFrame(rows, columns=columns)
