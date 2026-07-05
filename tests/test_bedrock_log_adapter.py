"""
test_bedrock_log_adapter.py
-----------------------------
Tests for src/bedrock_log_adapter.py: the JSON-to-CSV format transformer
that maps Bedrock-style trace/guardrail JSON into the internal graph
schema. These tests do not call AWS; they exercise the pure mapping logic
against the sample fixture and confirm the output is fully compatible
with the existing graph_builder / graph_optimizer / graph_metrics
pipeline, unmodified.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from bedrock_log_adapter import (
    load_bedrock_trace_file,
    adapt_bedrock_session,
    adapt_bedrock_sessions,
    TRACE_COLUMNS,
    EVENT_TYPE_TO_BOUNDARY,
    ACTION_TO_RESULT,
)
from graph_builder import build_all_workflow_graphs, PROTECTED_BOUNDARIES
from graph_optimizer import generate_optimized_plan, load_policy

SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_uploads", "sample_bedrock_trace.json"
)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "guardrail_policy.yaml")


@pytest.fixture(scope="module")
def adapted_df():
    return load_bedrock_trace_file(SAMPLE_PATH)


def test_sample_file_loads_without_error(adapted_df):
    assert isinstance(adapted_df, pd.DataFrame)
    assert not adapted_df.empty


def test_adapted_columns_match_internal_trace_schema(adapted_df):
    assert list(adapted_df.columns) == TRACE_COLUMNS


def test_no_raw_text_leaks_into_output_columns(adapted_df):
    # The adapter must only ever emit hashes, never the raw inputText/
    # outputText strings from the source trace.
    for col in ("text_hash", "input_text_hash", "output_text_hash"):
        assert adapted_df[col].str.len().eq(12).all()
        assert not adapted_df[col].str.contains(" ").any()


def test_event_type_boundary_mapping_is_applied(adapted_df):
    # PRE_PROCESSING -> USER_TO_AGENT and POST_PROCESSING -> AGENT_TO_FINAL_RESPONSE
    # are both present in the sample fixture.
    boundaries = set(adapted_df["trust_boundary"])
    assert "USER_TO_AGENT" in boundaries
    assert "AGENT_TO_FINAL_RESPONSE" in boundaries
    assert boundaries.issubset(set(EVENT_TYPE_TO_BOUNDARY.values()))


def test_guardrail_result_mapping(adapted_df):
    # The sample fixture includes a BLOCKED PII entity (PIN) on the
    # ACTION_GROUP_INVOCATION step, which must map to BLOCK.
    assert (adapted_df["guardrail_result"] == "BLOCK").any()
    assert set(adapted_df["guardrail_result"].unique()).issubset(set(ACTION_TO_RESULT.values()))


def test_explicit_risk_tag_is_respected(adapted_df):
    # The sample fixture tags the AccountLookupTool/SummarizerAgent hops
    # as high risk explicitly via riskTag.
    tool_rows = adapted_df[adapted_df["trust_boundary"] == "AGENT_TO_TOOL"]
    assert (tool_rows["risk_level"] == "high").all()


def test_workflow_and_request_id_derived_from_session(adapted_df):
    assert (adapted_df["workflow_id"] == "bedrock_AGENTDEMO123").all()
    assert (adapted_df["request_id"] == "sess-2026-07-05-000123").all()


def test_adapt_bedrock_sessions_handles_a_list_of_sessions():
    import json

    with open(SAMPLE_PATH) as f:
        single = json.load(f)

    combined = adapt_bedrock_sessions([single, single])
    single_df = adapt_bedrock_session(single)
    assert len(combined) == 2 * len(single_df)


def test_missing_guardrail_trace_produces_no_fabricated_check():
    # An event with no guardrailTrace at all must not invent a guardrail
    # check that was never actually evaluated.
    raw = {
        "agentId": "AGENTX",
        "sessionId": "sess-x",
        "traceEvents": [
            {
                "eventOrder": 1,
                "eventType": "ORCHESTRATION",
                "sourceNode": "User",
                "targetNode": "PlannerAgent",
                "inputText": "hello",
                "outputText": "hello",
            }
        ],
    }
    df = adapt_bedrock_session(raw)
    assert df.empty


def test_adapted_trace_builds_a_valid_workflow_graph(adapted_df):
    graphs = build_all_workflow_graphs(adapted_df)
    assert len(graphs) == 1
    wg = next(iter(graphs.values()))
    assert wg.user_node() is not None
    assert wg.final_nodes()
    paths = wg.all_paths_with_types()
    assert len(paths) >= 1
    assert paths[0][1] == "final"


def test_adapted_trace_runs_through_the_real_optimizer_unmodified(adapted_df):
    graphs = build_all_workflow_graphs(adapted_df)
    policy = load_policy(CONFIG_PATH)
    plan = generate_optimized_plan(graphs, policy)

    assert not plan.empty
    assert len(plan) == len(adapted_df)

    # Every protected-boundary row must be KEEP -- the same safety
    # invariant enforced for synthetic data must hold for adapted
    # real-shaped data too, since the optimizer code path is identical.
    protected_rows = plan[plan["trust_boundary"].isin(PROTECTED_BOUNDARIES)]
    assert (protected_rows["recommended_decision"] == "KEEP").all()

    # The sample fixture's duplicated internal PII check (same policy,
    # same text hash, medium risk) should be recognized as safe to
    # optimize rather than silently kept.
    internal_rows = plan[plan["trust_boundary"] == "AGENT_TO_AGENT_INTERNAL"]
    assert internal_rows["recommended_decision"].isin(
        ["REUSE_PREVIOUS_DECISION", "SKIP_REDUNDANT_WITH_AUDIT", "KEEP"]
    ).all()
    assert (internal_rows["recommended_decision"] == "REUSE_PREVIOUS_DECISION").any()
