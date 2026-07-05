"""
graph_features.py
------------------
Computes graph-level features for every workflow graph built by
graph_builder.py. This is what makes the optimizer "graph-first": features
describe path structure, boundary coverage, and redundancy shape across the
whole graph, not just row-by-row duplicate counts.

graph_risk_score uses transparent, rule-based weights (see
config/guardrail_policy.yaml -> risk_score_weights) rather than an opaque
ML model, so every score is fully explainable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from graph_builder import WorkflowGraph, PROTECTED_BOUNDARIES


def _duplicate_key_groups(checks: list) -> Dict[Tuple, int]:
    """Count how many times each (guardrail_type, policy_id, text_hash) key
    appears among a list of guardrail-check dict rows."""
    counts: Dict[Tuple, int] = {}
    for c in checks:
        key = (c.get("guardrail_type"), c.get("policy_id"), c.get("text_hash"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def compute_workflow_features(wg: WorkflowGraph, risk_weights: dict) -> Dict:
    """Compute the 19 graph-level features for a single WorkflowGraph."""
    node_count = len(wg.nodes)
    edge_count = len(wg.edges)

    paths = wg.all_paths()
    path_count = len(paths)
    max_path_length = max((len(p) - 1 for p in paths), default=0)

    protected_boundary_count = sum(1 for e in wg.edges.values() if e.trust_boundary in PROTECTED_BOUNDARIES)
    internal_handoff_count = sum(1 for e in wg.edges.values() if e.trust_boundary == "AGENT_TO_AGENT_INTERNAL")

    all_checks = []
    for e in wg.edges.values():
        all_checks.extend(e.guardrail_checks)

    total_guardrail_calls = len(all_checks)

    dup_counts = _duplicate_key_groups(all_checks)
    duplicate_guardrail_groups = sum(1 for v in dup_counts.values() if v > 1)

    # Redundant internal checks: extra occurrences (beyond the first) of a
    # (guardrail_type, policy_id, text_hash) key where the *duplicate*
    # occurrence itself sits on an optimizable (AGENT_TO_AGENT_INTERNAL)
    # edge. This must mirror graph_optimizer._decide_step's real "prev"
    # tracking, which follows a key across the WHOLE path -- including
    # protected boundaries -- and only asks whether the CURRENT (repeat)
    # check's edge is internal. USER_TO_AGENT is typically the first edge
    # on any path, so "protected boundary first, internal repeat second"
    # is the common real case, not an edge case: restricting the "first
    # occurrence" search to internal-only checks (as an earlier version of
    # this function did) silently missed nearly all of them and made
    # workflows_with_redundant_paths under-report versus the actual
    # optimizer plan. See tests/test_graph_features.py::
    # test_redundant_internal_checks_matches_optimizer_reuse_and_skip_decisions.
    key_first_seen: Dict[Tuple, dict] = {}
    redundant_internal_checks = 0
    for c in sorted(all_checks, key=lambda r: r.get("step_id", 0)):
        key = (c.get("guardrail_type"), c.get("policy_id"), c.get("text_hash"))
        if key not in key_first_seen:
            key_first_seen[key] = c
            continue
        edge = wg.edges.get((str(c.get("source_node")), str(c.get("target_node"))))
        if edge is not None and edge.trust_boundary == "AGENT_TO_AGENT_INTERNAL":
            redundant_internal_checks += 1

    high_risk_node_count = sum(1 for n in wg.nodes.values() if n.risk_level == "high")
    high_risk_edge_count = sum(1 for e in wg.edges.values() if e.edge_risk_level == "high")

    missing_final_boundary_check = wg.has_final_boundary_edge() and not wg.has_final_boundary_check()
    # Workflows lacking the edge entirely are also missing coverage.
    if not wg.has_final_boundary_edge():
        missing_final_boundary_check = True

    text_hashes = [c.get("text_hash") for c in all_checks]
    repeated_text_hash_count = len(text_hashes) - len(set(text_hashes)) if text_hashes else 0

    policy_drift_count = 0
    text_mutation_count = 0
    for e in wg.edges.values():
        checks_sorted = sorted(e.guardrail_checks, key=lambda r: r.get("step_id", 0))
        for c in checks_sorted:
            if c.get("text_changed"):
                text_mutation_count += 1
        policies = {c.get("policy_id") for c in checks_sorted}
        if len(policies) > 1:
            policy_drift_count += 1

    guardrail_density = round(total_guardrail_calls / edge_count, 3) if edge_count else 0.0
    boundary_guardrail_coverage = (
        round(protected_boundary_count / max(1, sum(1 for e in wg.edges.values())), 3)
    )
    internal_redundancy_ratio = (
        round(redundant_internal_checks / internal_handoff_count, 3) if internal_handoff_count else 0.0
    )

    graph_risk_score = 0
    for e in wg.edges.values():
        if e.trust_boundary != "AGENT_TO_AGENT_INTERNAL" and e.edge_risk_level == "high":
            graph_risk_score += risk_weights.get("high_risk_boundary", 3)
        if e.trust_boundary in ("AGENT_TO_TOOL", "TOOL_TO_AGENT"):
            graph_risk_score += risk_weights.get("tool_boundary", 2)
        for c in e.guardrail_checks:
            if c.get("guardrail_result") == "WARN":
                graph_risk_score += risk_weights.get("warn_result", 1)
            elif c.get("guardrail_result") == "BLOCK":
                graph_risk_score += risk_weights.get("block_result", 2)
    if missing_final_boundary_check:
        graph_risk_score += risk_weights.get("missing_final_check", 2)
    graph_risk_score += policy_drift_count * risk_weights.get("policy_drift", 1)
    graph_risk_score += text_mutation_count * risk_weights.get("text_mutation", 1)

    highest_redundancy_path = ""
    if paths:
        best_path, best_score = None, -1
        for p in paths:
            score = 0
            for i in range(len(p) - 1):
                edge = wg.edges.get((p[i], p[i + 1]))
                if edge:
                    score += max(0, len(edge.guardrail_checks) - 1)
            if score > best_score:
                best_score, best_path = score, p
        highest_redundancy_path = " -> ".join(best_path) if best_path else ""

    return {
        "workflow_id": wg.workflow_id,
        "request_id": wg.request_id,
        "node_count": node_count,
        "edge_count": edge_count,
        "path_count": path_count,
        "max_path_length": max_path_length,
        "protected_boundary_count": protected_boundary_count,
        "internal_handoff_count": internal_handoff_count,
        "total_guardrail_calls": total_guardrail_calls,
        "duplicate_guardrail_groups": duplicate_guardrail_groups,
        "redundant_internal_checks": redundant_internal_checks,
        "high_risk_node_count": high_risk_node_count,
        "high_risk_edge_count": high_risk_edge_count,
        "missing_final_boundary_check": missing_final_boundary_check,
        "repeated_text_hash_count": repeated_text_hash_count,
        "policy_drift_count": policy_drift_count,
        "text_mutation_count": text_mutation_count,
        "guardrail_density": guardrail_density,
        "boundary_guardrail_coverage": boundary_guardrail_coverage,
        "internal_redundancy_ratio": internal_redundancy_ratio,
        "graph_risk_score": graph_risk_score,
        "highest_redundancy_path": highest_redundancy_path,
    }


def compute_all_workflow_features(
    graphs: Dict[Tuple[str, str], WorkflowGraph], risk_weights: dict
) -> pd.DataFrame:
    rows = [compute_workflow_features(wg, risk_weights) for wg in graphs.values()]
    df = pd.DataFrame(rows)
    summary_cols = [
        "workflow_id",
        "request_id",
        "node_count",
        "edge_count",
        "max_path_length",
        "total_guardrail_calls",
        "duplicate_guardrail_groups",
        "redundant_internal_checks",
        "protected_boundary_count",
        "boundary_guardrail_coverage",
        "missing_final_boundary_check",
        "policy_drift_count",
        "text_mutation_count",
        "graph_risk_score",
        "highest_redundancy_path",
    ]
    # Keep the full feature set but ensure the required summary columns exist and are ordered first.
    other_cols = [c for c in df.columns if c not in summary_cols]
    return df[summary_cols + other_cols]


def save_workflow_graph_summary(
    graphs: Dict[Tuple[str, str], WorkflowGraph], risk_weights: dict, path: str
) -> pd.DataFrame:
    df = compute_all_workflow_features(graphs, risk_weights)
    df.to_csv(path, index=False)
    return df
