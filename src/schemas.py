"""
schemas.py
----------
Explicit, validated data models for the pipeline's stage boundaries
(Part 3.8): TraceRow, PlanRow, RedundantPathRow, GraphMetrics.

Uses `dataclasses` + explicit column-schema validation rather than passing
bare dicts/DataFrames between stages unchecked. This is deliberately the
"dataclasses + explicit validation" option called out as an acceptable
alternative to pydantic in the hardening brief -- it adds zero new runtime
dependencies while still failing fast with a clear error if the synthetic
CSV schema ever drifts, instead of silently producing wrong output.

Usage:
    from schemas import validate_trace_df, validate_plan_df

    df = pd.read_csv(path)
    validate_trace_df(df)   # raises SchemaValidationError on drift
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when a DataFrame's columns/dtypes don't match the expected schema."""


@dataclass(frozen=True)
class TraceRow:
    """One row of the raw synthetic guardrail-check trace
    (data/synthetic_workflow_traces.csv)."""

    workflow_id: str
    request_id: str
    step_id: int
    source_node: str
    target_node: str
    trust_boundary: str
    guardrail_type: str
    policy_id: str
    text_hash: str
    text_changed: bool
    guardrail_result: str
    risk_level: str
    timestamp: str
    behavior_pattern: str


@dataclass(frozen=True)
class PlanRow:
    """One row of the optimized guardrail plan
    (outputs/optimized_guardrail_plan.csv)."""

    workflow_id: str
    request_id: str
    step_id: int
    guardrail_type: str
    trust_boundary: str
    recommended_decision: str
    reason: str
    reused_from_step_id: Optional[int]
    safe_to_optimize: bool
    graph_path_id: str
    path_type: str
    protected_boundary: bool
    audit_required: bool


@dataclass(frozen=True)
class RedundantPathRow:
    """One row of the graph-path-aware redundant-check report
    (outputs/redundant_paths.csv)."""

    workflow_id: str
    request_id: str
    path_id: str
    path_type: str
    path_nodes: str
    guardrail_type: str
    policy_id: str
    text_hash: str
    first_step_id: int
    duplicate_step_id: int
    duplicate_reason: str
    optimization_allowed: bool
    blocked_by: str
    recommended_decision: str


@dataclass(frozen=True)
class GraphMetrics:
    """Aggregate metrics block (outputs/guardrail_metrics.json)."""

    original_guardrail_calls: int
    optimized_guardrail_calls: int
    guardrail_call_reduction_percent: float
    latency_saved_percent: float
    cost_saved_percent: float
    boundary_coverage_preserved_percent: float
    high_risk_preservation_percent: float
    human_review_count: int


def _expected_columns(model_cls) -> List[str]:
    return [f.name for f in fields(model_cls)]


def _validate_columns(df: pd.DataFrame, model_cls, df_name: str) -> None:
    expected = set(_expected_columns(model_cls))
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise SchemaValidationError(
            f"{df_name} is missing expected column(s) {sorted(missing)}. "
            f"This usually means the synthetic data generator or an upstream "
            f"pipeline stage schema has drifted out of sync with {model_cls.__name__}. "
            f"Expected columns: {sorted(expected)}. Got: {sorted(actual)}."
        )
    if df.empty:
        raise SchemaValidationError(f"{df_name} has zero rows -- expected at least one row of data.")


def validate_trace_df(df: pd.DataFrame) -> None:
    """Fail fast if data/synthetic_workflow_traces.csv drifts from TraceRow."""
    _validate_columns(df, TraceRow, "Synthetic trace DataFrame")


def validate_plan_df(df: pd.DataFrame) -> None:
    """Fail fast if outputs/optimized_guardrail_plan.csv drifts from PlanRow."""
    _validate_columns(df, PlanRow, "Optimized plan DataFrame")


def validate_redundant_paths_df(df: pd.DataFrame) -> None:
    """Fail fast if outputs/redundant_paths.csv drifts from RedundantPathRow."""
    _validate_columns(df, RedundantPathRow, "Redundant paths DataFrame")


def validate_metrics_dict(metrics: Dict[str, Any]) -> None:
    """Fail fast if the 'overall'+'safety' metrics dict drifts from GraphMetrics."""
    flat = {**metrics.get("overall", {}), **metrics.get("safety", {})}
    expected = set(_expected_columns(GraphMetrics))
    missing = expected - set(flat.keys())
    if missing:
        raise SchemaValidationError(
            f"Metrics dict is missing expected key(s) {sorted(missing)} "
            f"(GraphMetrics schema). Got keys: {sorted(flat.keys())}."
        )
