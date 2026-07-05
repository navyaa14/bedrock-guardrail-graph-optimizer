"""
test_schemas.py
-----------------
Tests for schemas.py -- the explicit, validated data models for the
pipeline's stage boundaries (TraceRow, PlanRow, RedundantPathRow,
GraphMetrics). This project uses `dataclasses` + explicit column-schema
validation as its "pydantic-equivalent" data-validation layer (see the
module docstring in src/schemas.py): it adds zero new runtime dependencies
while still failing fast, with a clear error, if the synthetic CSV schema
or a downstream stage's DataFrame ever drifts out of sync with the
expected shape.

These tests prove that fail-fast behavior actually fires on real drift
(missing columns, empty DataFrames, missing metrics keys) and does not
false-positive on well-formed pipeline output.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from synthetic_workflows import generate_synthetic_workflows
from graph_builder import build_all_workflow_graphs
from graph_optimizer import load_policy, generate_optimized_plan, detect_redundant_paths
from graph_features import compute_all_workflow_features
from graph_metrics import compute_metrics
from schemas import (
    SchemaValidationError,
    TraceRow,
    PlanRow,
    RedundantPathRow,
    GraphMetrics,
    validate_trace_df,
    validate_plan_df,
    validate_redundant_paths_df,
    validate_metrics_dict,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "guardrail_policy.yaml")


@pytest.fixture(scope="module")
def policy():
    return load_policy(CONFIG_PATH)


@pytest.fixture(scope="module")
def df():
    return generate_synthetic_workflows(n_workflows=60, seed=13)


@pytest.fixture(scope="module")
def graphs(df):
    return build_all_workflow_graphs(df)


@pytest.fixture(scope="module")
def plan_df(graphs, policy):
    return generate_optimized_plan(graphs, policy, strict_mode=False)


@pytest.fixture(scope="module")
def redundant_df(graphs, policy):
    return detect_redundant_paths(graphs, policy)


@pytest.fixture(scope="module")
def feature_df(graphs, policy):
    return compute_all_workflow_features(graphs, policy.get("risk_score_weights", {}))


@pytest.fixture(scope="module")
def metrics(df, graphs, plan_df, feature_df, policy):
    return compute_metrics(df, graphs, plan_df, feature_df, policy)


# ---------------------------------------------------------------------------
# Happy-path: real pipeline output should always validate cleanly.
# ---------------------------------------------------------------------------


def test_validate_trace_df_passes_for_well_formed_synthetic_data(df):
    # Should not raise.
    validate_trace_df(df)


def test_validate_plan_df_passes_for_generated_plan(plan_df):
    validate_plan_df(plan_df)


def test_validate_redundant_paths_df_passes_when_nonempty(redundant_df):
    assert len(redundant_df) > 0
    validate_redundant_paths_df(redundant_df)


def test_validate_metrics_dict_passes_for_well_formed_metrics(metrics):
    validate_metrics_dict(metrics)


# ---------------------------------------------------------------------------
# Fail-fast: schema drift must be caught, not silently ignored.
# ---------------------------------------------------------------------------


def test_validate_trace_df_raises_on_missing_column(df):
    tampered = df.drop(columns=["trust_boundary"])
    with pytest.raises(SchemaValidationError):
        validate_trace_df(tampered)


def test_validate_trace_df_raises_on_empty_dataframe(df):
    empty = df.iloc[0:0]
    with pytest.raises(SchemaValidationError):
        validate_trace_df(empty)


def test_validate_plan_df_raises_on_missing_column(plan_df):
    tampered = plan_df.drop(columns=["audit_required"])
    with pytest.raises(SchemaValidationError):
        validate_plan_df(tampered)


def test_validate_redundant_paths_df_raises_on_missing_column(redundant_df):
    tampered = redundant_df.drop(columns=["recommended_decision"])
    with pytest.raises(SchemaValidationError):
        validate_redundant_paths_df(tampered)


def test_validate_metrics_dict_raises_on_missing_key(metrics):
    tampered = {
        "overall": {k: v for k, v in metrics["overall"].items() if k != "cost_saved_percent"},
        "safety": dict(metrics["safety"]),
    }
    with pytest.raises(SchemaValidationError):
        validate_metrics_dict(tampered)


def test_validate_trace_df_error_message_names_missing_columns(df):
    tampered = df.drop(columns=["risk_level", "guardrail_result"])
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_trace_df(tampered)
    message = str(exc_info.value)
    assert "risk_level" in message
    assert "guardrail_result" in message


# ---------------------------------------------------------------------------
# Schema definitions themselves must include the columns every downstream
# stage actually depends on (prevents someone quietly deleting a dataclass
# field and having validation silently stop protecting that column).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected_field",
    ["reused_from_step_id", "audit_required", "protected_boundary", "recommended_decision", "step_id"],
)
def test_plan_row_schema_includes_safety_critical_fields(expected_field):
    field_names = set(PlanRow.__dataclass_fields__)
    assert expected_field in field_names


def test_trace_row_schema_includes_core_graph_fields():
    field_names = set(TraceRow.__dataclass_fields__)
    for expected in ("workflow_id", "request_id", "step_id", "trust_boundary", "guardrail_type"):
        assert expected in field_names


def test_graph_metrics_schema_includes_hard_safety_gate_fields():
    field_names = set(GraphMetrics.__dataclass_fields__)
    assert "boundary_coverage_preserved_percent" in field_names
    assert "high_risk_preservation_percent" in field_names


def test_redundant_path_row_schema_includes_blocking_fields():
    field_names = set(RedundantPathRow.__dataclass_fields__)
    assert "blocked_by" in field_names
    assert "optimization_allowed" in field_names
