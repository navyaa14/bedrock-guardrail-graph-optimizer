"""
graph_metrics.py
-----------------
Computes overall, graph-level, and safety metrics from the optimized
guardrail plan, plus the trust-boundary coverage matrix.

Hard safety requirement (enforced, not just reported): protected
trust-boundary coverage and high-risk check preservation must both be
100%. If either drops below 100%, this module raises a RuntimeError so the
pipeline fails loudly rather than silently shipping an unsafe plan.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from graph_builder import WorkflowGraph, PROTECTED_BOUNDARIES

# Simulated cost/latency for a newly-added final-boundary check
# (MOVE_TO_BOUNDARY rows have no original execution to compare against).
# Calibrated to the same GROUNDING_CHECK-class numbers as
# synthetic_workflows.py._latency_and_cost (a final-response boundary
# check is most analogous to a contextual grounding check) -- see the
# pricing/latency calibration comment block in config/guardrail_policy.yaml.
SIMULATED_BOUNDARY_CHECK_LATENCY_MS = 260.0
SIMULATED_BOUNDARY_CHECK_COST_USD = 0.0003

NON_SKIPPED_DECISIONS = {"KEEP", "HUMAN_REVIEW_REQUIRED", "REUSE_PREVIOUS_DECISION", "MOVE_TO_BOUNDARY"}


def _optimized_latency_cost(row, savings_model: dict) -> Tuple[float, float]:
    decision = row["recommended_decision"]
    if decision == "MOVE_TO_BOUNDARY" and row["step_id"] == -1:
        return SIMULATED_BOUNDARY_CHECK_LATENCY_MS, SIMULATED_BOUNDARY_CHECK_COST_USD
    savings_pct = savings_model.get(decision, 0.0)
    latency = float(row["original_latency_ms"]) * (1 - savings_pct)
    cost = float(row["original_cost_usd"]) * (1 - savings_pct)
    return latency, cost


def compute_metrics(
    df: pd.DataFrame,
    graphs: Dict[Tuple[str, str], WorkflowGraph],
    plan_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    policy: dict,
) -> dict:
    savings_model = policy.get(
        "savings_model",
        {
            "KEEP": 0.0,
            "HUMAN_REVIEW_REQUIRED": 0.0,
            "MOVE_TO_BOUNDARY": 0.0,
            "REUSE_PREVIOUS_DECISION": 0.80,
            "SKIP_REDUNDANT_WITH_AUDIT": 1.00,
        },
    )
    protected = set(policy["protected_boundaries"])

    real_rows = plan_df[plan_df["step_id"] != -1].copy()
    move_rows = plan_df[plan_df["step_id"] == -1].copy()

    opt_latency_cost = plan_df.apply(lambda r: _optimized_latency_cost(r, savings_model), axis=1)
    plan_df = plan_df.copy()
    plan_df["optimized_latency_ms"] = [x[0] for x in opt_latency_cost]
    plan_df["optimized_cost_usd"] = [x[1] for x in opt_latency_cost]

    # ---------------- Overall ----------------
    original_guardrail_calls = int(len(df))
    optimized_guardrail_calls = int(
        len(real_rows[real_rows["recommended_decision"] != "SKIP_REDUNDANT_WITH_AUDIT"]) + len(move_rows)
    )
    guardrail_call_reduction_percent = (
        round(100 * (1 - optimized_guardrail_calls / original_guardrail_calls), 2)
        if original_guardrail_calls
        else 0.0
    )

    original_total_latency_ms = float(df["latency_ms"].sum())
    optimized_total_latency_ms = float(plan_df["optimized_latency_ms"].sum())
    # Derive the "saved" figure from the *same rounded totals* reported
    # alongside it (round(a) - round(b)) rather than rounding the raw
    # difference separately (round(a - b)). The two can differ by the
    # reporting precision (e.g. 0.2139 - 0.1946 = 0.0193 but the unrounded
    # difference rounds to 0.0192), which breaks the invariant that
    # saved == original_total - optimized_total once both are rounded for
    # display/JSON output.
    latency_saved_ms = round(round(original_total_latency_ms, 2) - round(optimized_total_latency_ms, 2), 2)
    latency_saved_percent = (
        round(100 * latency_saved_ms / original_total_latency_ms, 2) if original_total_latency_ms else 0.0
    )

    original_total_cost_usd = float(df["estimated_cost_usd"].sum())
    optimized_total_cost_usd = float(plan_df["optimized_cost_usd"].sum())
    cost_saved_usd = round(round(original_total_cost_usd, 4) - round(optimized_total_cost_usd, 4), 4)
    cost_saved_percent = (
        round(100 * cost_saved_usd / original_total_cost_usd, 2) if original_total_cost_usd else 0.0
    )

    # ---------------- Graph-level ----------------
    workflows_analyzed = int(feature_df["workflow_id"].nunique())
    average_nodes_per_workflow = round(float(feature_df["node_count"].mean()), 2)
    average_edges_per_workflow = round(float(feature_df["edge_count"].mean()), 2)
    average_path_length = round(float(feature_df["max_path_length"].mean()), 2)

    workflows_with_redundant_paths = int((feature_df["redundant_internal_checks"] > 0).sum())

    total_redundant_paths_detected = int(
        len(real_rows[real_rows["reused_from_step_id"].notna()])
    )
    redundant_paths_optimized = int(
        len(
            real_rows[
                real_rows["recommended_decision"].isin(["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"])
            ]
        )
    )
    redundant_paths_blocked_by_safety = int(
        len(
            real_rows[
                (real_rows["reused_from_step_id"].notna())
                & (real_rows["recommended_decision"].isin(["KEEP", "HUMAN_REVIEW_REQUIRED"]))
            ]
        )
    )
    graph_risk_score_average = round(float(feature_df["graph_risk_score"].mean()), 2)

    # ---------------- Safety ----------------
    boundary_checks_original = int((df["trust_boundary"].isin(protected)).sum())
    protected_real_rows = real_rows[real_rows["trust_boundary"].isin(protected)]
    boundary_checks_preserved = int((protected_real_rows["recommended_decision"] == "KEEP").sum())
    boundary_coverage_preserved_percent = (
        round(100 * boundary_checks_preserved / boundary_checks_original, 2) if boundary_checks_original else 100.0
    )

    high_risk_original_rows = df[(df["risk_level"] == "high") | (df.get("edge_risk_level", df["risk_level"]) == "high")]
    high_risk_checks_original = int(len(high_risk_original_rows))
    high_risk_real_rows = real_rows[
        (real_rows["risk_level"] == "high")
    ]
    high_risk_checks_preserved = int(
        (high_risk_real_rows["recommended_decision"].isin(["KEEP", "HUMAN_REVIEW_REQUIRED"])).sum()
    )
    high_risk_preservation_percent = (
        round(100 * high_risk_checks_preserved / high_risk_checks_original, 2) if high_risk_checks_original else 100.0
    )

    false_skip_rate_simulated = round(
        100
        * len(
            real_rows[
                (real_rows["recommended_decision"] == "SKIP_REDUNDANT_WITH_AUDIT")
                & (real_rows["trust_boundary"].isin(protected))
            ]
        )
        / max(1, len(real_rows)),
        4,
    )

    policy_drift_blocks = int((real_rows["reason"] == "policy_drift_since_previous_check").sum())
    text_mutation_blocks = int((real_rows["reason"] == "text_changed_since_previous_check").sum())
    warn_block_escalations = int(
        real_rows["reason"].str.contains("previous_result_(?:WARN|BLOCK)", regex=True, na=False).sum()
    )
    human_review_count = int((plan_df["recommended_decision"] == "HUMAN_REVIEW_REQUIRED").sum())
    audit_required_count = int((plan_df["audit_required"] == True).sum())  # noqa: E712

    metrics = {
        "overall": {
            "original_guardrail_calls": original_guardrail_calls,
            "optimized_guardrail_calls": optimized_guardrail_calls,
            "guardrail_call_reduction_percent": guardrail_call_reduction_percent,
            "original_total_latency_ms": round(original_total_latency_ms, 2),
            "optimized_total_latency_ms": round(optimized_total_latency_ms, 2),
            "latency_saved_ms": round(latency_saved_ms, 2),
            "latency_saved_percent": latency_saved_percent,
            "original_total_cost_usd": round(original_total_cost_usd, 4),
            "optimized_total_cost_usd": round(optimized_total_cost_usd, 4),
            "cost_saved_usd": round(cost_saved_usd, 4),
            "cost_saved_percent": cost_saved_percent,
        },
        "graph": {
            "workflows_analyzed": workflows_analyzed,
            "average_nodes_per_workflow": average_nodes_per_workflow,
            "average_edges_per_workflow": average_edges_per_workflow,
            "average_path_length": average_path_length,
            "workflows_with_redundant_paths": workflows_with_redundant_paths,
            "total_redundant_paths_detected": total_redundant_paths_detected,
            "redundant_paths_optimized": redundant_paths_optimized,
            "redundant_paths_blocked_by_safety": redundant_paths_blocked_by_safety,
            "graph_risk_score_average": graph_risk_score_average,
        },
        "safety": {
            "boundary_checks_original": boundary_checks_original,
            "boundary_checks_preserved": boundary_checks_preserved,
            "boundary_coverage_preserved_percent": boundary_coverage_preserved_percent,
            "high_risk_checks_original": high_risk_checks_original,
            "high_risk_checks_preserved": high_risk_checks_preserved,
            "high_risk_preservation_percent": high_risk_preservation_percent,
            "false_skip_rate_simulated": false_skip_rate_simulated,
            "policy_drift_blocks": policy_drift_blocks,
            "text_mutation_blocks": text_mutation_blocks,
            "warn_block_escalations": warn_block_escalations,
            "human_review_count": human_review_count,
            "audit_required_count": audit_required_count,
        },
    }

    # ---------------- Hard safety gate ----------------
    if boundary_coverage_preserved_percent < 100.0:
        raise RuntimeError(
            f"SAFETY VIOLATION: protected trust-boundary coverage is "
            f"{boundary_coverage_preserved_percent}%, must be 100%."
        )
    if high_risk_preservation_percent < 100.0:
        raise RuntimeError(
            f"SAFETY VIOLATION: high-risk check preservation is "
            f"{high_risk_preservation_percent}%, must be 100%."
        )

    return metrics


def compute_scenario_metrics(df: pd.DataFrame, plan_df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Compute the same call-reduction / latency / cost / safety metrics as
    `compute_metrics`, but broken out per synthetic scenario (workflow
    template) instead of aggregated across the whole corpus.

    This gives a reproducible, auditable breakdown of exactly which
    scenario archetypes (long_chain, healthcare_triage, malformed_workflow,
    etc.) contribute how much of the overall savings and confirms safety
    invariants (boundary coverage, high-risk preservation) hold
    independently within *every* scenario, not just in aggregate.

    Written to outputs/scenario_metrics.csv by run_pipeline.py.
    """
    savings_model = policy.get(
        "savings_model",
        {
            "KEEP": 0.0,
            "HUMAN_REVIEW_REQUIRED": 0.0,
            "MOVE_TO_BOUNDARY": 0.0,
            "REUSE_PREVIOUS_DECISION": 0.80,
            "SKIP_REDUNDANT_WITH_AUDIT": 1.00,
        },
    )
    protected = set(policy["protected_boundaries"])

    # Each workflow_id is generated from exactly one template/scenario, so a
    # simple first() groupby gives a clean workflow_id -> scenario mapping.
    workflow_scenario = df.groupby("workflow_id")["workflow_template"].first()

    rows = []
    for scenario in sorted(workflow_scenario.unique()):
        wf_ids = set(workflow_scenario[workflow_scenario == scenario].index)

        df_s = df[df["workflow_id"].isin(wf_ids)]
        plan_s = plan_df[plan_df["workflow_id"].isin(wf_ids)].copy()

        real_rows = plan_s[plan_s["step_id"] != -1].copy()
        move_rows = plan_s[plan_s["step_id"] == -1].copy()

        if len(plan_s):
            opt_latency_cost = plan_s.apply(lambda r: _optimized_latency_cost(r, savings_model), axis=1)
            plan_s["optimized_latency_ms"] = [x[0] for x in opt_latency_cost]
            plan_s["optimized_cost_usd"] = [x[1] for x in opt_latency_cost]
        else:
            plan_s["optimized_latency_ms"] = []
            plan_s["optimized_cost_usd"] = []

        original_guardrail_calls = int(len(df_s))
        optimized_guardrail_calls = int(
            len(real_rows[real_rows["recommended_decision"] != "SKIP_REDUNDANT_WITH_AUDIT"]) + len(move_rows)
        )
        call_reduction_percent = (
            round(100 * (1 - optimized_guardrail_calls / original_guardrail_calls), 2)
            if original_guardrail_calls
            else 0.0
        )

        original_total_latency_ms = float(df_s["latency_ms"].sum())
        optimized_total_latency_ms = float(plan_s["optimized_latency_ms"].sum())
        latency_saved_ms = round(round(original_total_latency_ms, 2) - round(optimized_total_latency_ms, 2), 2)
        latency_saved_percent = (
            round(100 * latency_saved_ms / original_total_latency_ms, 2) if original_total_latency_ms else 0.0
        )

        original_total_cost_usd = float(df_s["estimated_cost_usd"].sum())
        optimized_total_cost_usd = float(plan_s["optimized_cost_usd"].sum())
        cost_saved_usd = round(round(original_total_cost_usd, 4) - round(optimized_total_cost_usd, 4), 4)
        cost_saved_percent = (
            round(100 * cost_saved_usd / original_total_cost_usd, 2) if original_total_cost_usd else 0.0
        )

        boundary_checks_original = int((df_s["trust_boundary"].isin(protected)).sum())
        protected_real_rows = real_rows[real_rows["trust_boundary"].isin(protected)]
        boundary_checks_preserved = int((protected_real_rows["recommended_decision"] == "KEEP").sum())
        boundary_coverage_percent = (
            round(100 * boundary_checks_preserved / boundary_checks_original, 2)
            if boundary_checks_original
            else 100.0
        )

        high_risk_original_rows = df_s[
            (df_s["risk_level"] == "high") | (df_s.get("edge_risk_level", df_s["risk_level"]) == "high")
        ]
        high_risk_checks_original = int(len(high_risk_original_rows))
        high_risk_real_rows = real_rows[real_rows["risk_level"] == "high"]
        high_risk_checks_preserved = int(
            (high_risk_real_rows["recommended_decision"].isin(["KEEP", "HUMAN_REVIEW_REQUIRED"])).sum()
        )
        high_risk_preservation_percent = (
            round(100 * high_risk_checks_preserved / high_risk_checks_original, 2)
            if high_risk_checks_original
            else 100.0
        )

        rows.append(
            {
                "scenario": scenario,
                "workflow_count": int(len(wf_ids)),
                "original_guardrail_calls": original_guardrail_calls,
                "optimized_guardrail_calls": optimized_guardrail_calls,
                "call_reduction_percent": call_reduction_percent,
                "latency_saved_percent": latency_saved_percent,
                "cost_saved_percent": cost_saved_percent,
                "boundary_coverage_percent": boundary_coverage_percent,
                "high_risk_preservation_percent": high_risk_preservation_percent,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "workflow_count",
            "original_guardrail_calls",
            "optimized_guardrail_calls",
            "call_reduction_percent",
            "latency_saved_percent",
            "cost_saved_percent",
            "boundary_coverage_percent",
            "high_risk_preservation_percent",
        ],
    )


def compute_boundary_coverage_matrix(plan_df: pd.DataFrame, df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Build outputs/boundary_coverage_matrix.csv."""
    protected = set(policy["protected_boundaries"])
    real_rows = plan_df[plan_df["step_id"] != -1]

    rows = []
    for boundary in sorted(set(df["trust_boundary"].unique()) | protected):
        original_checks = int((df["trust_boundary"] == boundary).sum())
        boundary_rows = real_rows[real_rows["trust_boundary"] == boundary]
        preserved_checks = int((boundary_rows["recommended_decision"] == "KEEP").sum())
        if boundary in protected:
            # Protected boundaries: coverage counts KEEP + HUMAN_REVIEW_REQUIRED
            # (both mean the check is not silently skipped).
            preserved_checks = int(
                boundary_rows["recommended_decision"].isin(["KEEP", "HUMAN_REVIEW_REQUIRED"]).sum()
            )
        coverage_percent = round(100 * preserved_checks / original_checks, 2) if original_checks else 100.0
        optimized_checks = int(
            boundary_rows["recommended_decision"].isin(["SKIP_REDUNDANT_WITH_AUDIT", "REUSE_PREVIOUS_DECISION"]).sum()
        )
        human_review_checks = int((boundary_rows["recommended_decision"] == "HUMAN_REVIEW_REQUIRED").sum())

        rows.append(
            {
                "trust_boundary": boundary,
                "original_checks": original_checks,
                "preserved_checks": preserved_checks,
                "coverage_percent": coverage_percent,
                "optimized_checks": optimized_checks,
                "human_review_checks": human_review_checks,
            }
        )

    matrix = pd.DataFrame(rows)

    for boundary in protected:
        boundary_row = matrix[matrix["trust_boundary"] == boundary]
        if not boundary_row.empty and boundary_row.iloc[0]["coverage_percent"] < 100.0:
            raise RuntimeError(
                f"SAFETY VIOLATION: protected boundary {boundary} coverage is "
                f"{boundary_row.iloc[0]['coverage_percent']}%, must be 100%."
            )

    return matrix
