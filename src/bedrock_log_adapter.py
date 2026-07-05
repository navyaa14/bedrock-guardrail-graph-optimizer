"""
bedrock_log_adapter.py
------------------------
Adapter interface that normalizes real Bedrock-style trace/guardrail JSON
into this project's internal graph schema (the same columns produced by
synthetic_workflows.py), so the existing graph_builder / graph_optimizer /
graph_metrics pipeline can run unmodified against it.

IMPORTANT -- what this is, and what it is not:

  - This module does NOT call any live AWS API. It does not use boto3 and
    it does not fetch anything from your AWS account. It is a pure
    format-transformer: JSON in, DataFrame/CSV out.
  - There is no public dataset of real Bedrock production traces to
    download. Bedrock Agent traces and Guardrail assessments are
    account-specific and only exist once you invoke your own Bedrock
    Agents / Guardrails with logging enabled. See:
      - InvokeAgent trace (enableTrace=true):
        https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeAgent.html
      - Model invocation logging (CloudWatch/S3):
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html
      - ApplyGuardrail assessments:
        https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
  - This adapter expects you to have already exported such a trace/log to
    a JSON file (e.g. by capturing an InvokeAgent response stream with
    enableTrace=true, or a CloudWatch Logs export). A realistic example
    shape is provided in sample_uploads/sample_bedrock_trace.json.
  - Field mappings below are deliberately explicit and conservative. Real
    Bedrock traces do NOT contain a "trust boundary" or a business
    "risk_level" -- those are this project's concepts. This adapter
    derives trust_boundary from the trace event type (which IS present in
    real traces), and falls back to a keyword heuristic for risk_level
    only when the input JSON doesn't supply an explicit riskTag. That
    heuristic is clearly marked below and should not be treated as a
    substitute for a real business risk classification.
  - Guardrail confidence/severity labels (NONE/LOW/MEDIUM/HIGH) are
    mapped to approximate numeric values for internal consistency with
    the synthetic schema. These are approximations, not calibrated
    probabilities.
  - Cost is not returned by Bedrock trace/guardrail APIs in a directly
    usable per-check form. estimated_cost_usd here is a placeholder
    computed from a configurable per-unit constant; replace
    COST_PER_GUARDRAIL_UNIT_USD with your account's actual Bedrock
    Guardrails pricing before trusting any cost figure derived from it.

Usage:
    python src/bedrock_log_adapter.py \\
        --input sample_uploads/sample_bedrock_trace.json \\
        --out data/real_bedrock_trace.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, List, Optional

import pandas as pd

# Mirrors the row schema produced by synthetic_workflows.py /
# data/synthetic_workflow_traces.csv in this project. Kept as an explicit
# local constant (rather than imported) so this adapter has no import-time
# dependency on the synthetic generator's internals -- only the shared
# on-disk CSV contract.
TRACE_COLUMNS = [
    "workflow_id",
    "request_id",
    "step_id",
    "node_id",
    "source_node",
    "target_node",
    "node_type",
    "step_type",
    "agent_name",
    "guardrail_type",
    "policy_id",
    "text_hash",
    "input_text_hash",
    "output_text_hash",
    "text_changed",
    "trust_boundary",
    "risk_level",
    "edge_risk_level",
    "guardrail_result",
    "severity",
    "confidence",
    "latency_ms",
    "estimated_cost_usd",
    "timestamp",
    "workflow_template",
    "behavior_pattern",
]

# ---------------------------------------------------------------------------
# Mapping tables -- these encode the actual AWS trace/guardrail vocabulary.
# ---------------------------------------------------------------------------

# Bedrock Agent trace event types (InvokeAgent, enableTrace=true) mapped to
# this project's trust-boundary vocabulary.
EVENT_TYPE_TO_BOUNDARY = {
    "PRE_PROCESSING": "USER_TO_AGENT",
    "ORCHESTRATION": "AGENT_TO_AGENT_INTERNAL",
    "ACTION_GROUP_INVOCATION": "AGENT_TO_TOOL",
    "ACTION_GROUP_RESPONSE": "TOOL_TO_AGENT",
    "KNOWLEDGE_BASE_LOOKUP": "AGENT_TO_TOOL",
    "KNOWLEDGE_BASE_RESPONSE": "TOOL_TO_AGENT",
    "POST_PROCESSING": "AGENT_TO_FINAL_RESPONSE",
}

# Node type follows directly from the event type: action-group and
# knowledge-base steps are tool boundaries, everything else is an agent
# hop, and POST_PROCESSING always terminates at the final response node.
EVENT_TYPE_TO_NODE_TYPE = {
    "PRE_PROCESSING": "agent",
    "ORCHESTRATION": "agent",
    "ACTION_GROUP_INVOCATION": "tool",
    "ACTION_GROUP_RESPONSE": "agent",
    "KNOWLEDGE_BASE_LOOKUP": "tool",
    "KNOWLEDGE_BASE_RESPONSE": "agent",
    "POST_PROCESSING": "final",
}

# ApplyGuardrail contentPolicy.filters[].type values mapped onto this
# project's guardrail_type vocabulary. Real Bedrock Guardrails filter
# types: SEXUAL, VIOLENCE, HATE, INSULTS, MISCONDUCT, PROMPT_ATTACK.
CONTENT_FILTER_TYPE_TO_GUARDRAIL_TYPE = {
    "PROMPT_ATTACK": "PROMPT_ATTACK_CHECK",
    "SEXUAL": "TOXICITY_CHECK",
    "VIOLENCE": "TOXICITY_CHECK",
    "HATE": "TOXICITY_CHECK",
    "INSULTS": "TOXICITY_CHECK",
    "MISCONDUCT": "TOXICITY_CHECK",
}

# Confidence/filter-strength labels used throughout Guardrails assessments,
# mapped to an approximate numeric confidence for internal consistency.
# These are NOT calibrated probabilities -- they are ordinal placeholders.
CONFIDENCE_LABEL_TO_FLOAT = {
    "NONE": 0.05,
    "LOW": 0.4,
    "MEDIUM": 0.65,
    "HIGH": 0.92,
}

# Guardrail assessment actions mapped to this project's PASS/WARN/BLOCK
# result vocabulary.
ACTION_TO_RESULT = {
    "NONE": "PASS",
    "ANONYMIZED": "WARN",
    "MASKED": "WARN",
    "BLOCKED": "BLOCK",
    "GUARDRAIL_INTERVENED": "BLOCK",
    "INTERVENED": "BLOCK",
}

# Fallback only: used when the input JSON doesn't supply an explicit
# riskTag on a node/event. This is a coarse keyword heuristic, not a
# business risk classification -- real deployments should supply
# riskTag explicitly (e.g. from an agent/tool registry) instead of
# relying on this default.
RISK_KEYWORD_DEFAULTS = {
    "finance": "high",
    "payment": "high",
    "health": "high",
    "medical": "high",
    "compliance": "medium",
}

# Placeholder per-guardrail-unit cost. Bedrock Guardrails pricing is
# billed per text unit processed, not returned inline on each assessment;
# replace this with your account's actual per-unit rate before treating
# any derived cost figure as accurate.
COST_PER_GUARDRAIL_UNIT_USD = 0.000075


def _hash_text(text: Optional[str]) -> str:
    """Hash raw trace text locally so no raw customer content is ever
    written into the internal CSV schema (which stores hashes only)."""
    if not text:
        return hashlib.sha256(b"").hexdigest()[:12]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _infer_risk_level(explicit_risk: Optional[str], *names: str) -> str:
    if explicit_risk:
        return explicit_risk
    haystack = " ".join(n.lower() for n in names if n)
    for keyword, level in RISK_KEYWORD_DEFAULTS.items():
        if keyword in haystack:
            return level
    return "low"


def _iter_assessment_checks(assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand one ApplyGuardrail-style `assessment` object into a list of
    individual guardrail-check dicts, one per policy filter/entity found.

    Real assessments can contain multiple policy blocks at once
    (contentPolicy, sensitiveInformationPolicy, wordPolicy,
    contextualGroundingPolicy); each one becomes its own guardrail_type
    row in the internal schema, matching how the synthetic generator
    treats each guardrail_type as an independent check.
    """
    checks: List[Dict[str, Any]] = []

    content_policy = assessment.get("contentPolicy", {})
    for f in content_policy.get("filters", []):
        filter_type = f.get("type", "")
        checks.append({
            "guardrail_type": CONTENT_FILTER_TYPE_TO_GUARDRAIL_TYPE.get(filter_type, "TOXICITY_CHECK"),
            "result": ACTION_TO_RESULT.get(f.get("action", "NONE"), "PASS"),
            "confidence": CONFIDENCE_LABEL_TO_FLOAT.get(f.get("confidence", "NONE"), 0.05),
            "severity_label": f.get("filterStrength", f.get("confidence", "NONE")),
        })

    sensitive_policy = assessment.get("sensitiveInformationPolicy", {})
    pii_entities = sensitive_policy.get("piiEntities", [])
    if pii_entities:
        worst_action = "NONE"
        for e in pii_entities:
            if e.get("action") in ("BLOCKED", "GUARDRAIL_INTERVENED"):
                worst_action = "BLOCKED"
                break
            if e.get("action") in ("ANONYMIZED", "MASKED"):
                worst_action = "ANONYMIZED"
        checks.append({
            "guardrail_type": "PII_CHECK",
            "result": ACTION_TO_RESULT.get(worst_action, "PASS"),
            "confidence": 0.9 if pii_entities else 0.05,
            "severity_label": "HIGH" if worst_action == "BLOCKED" else ("MEDIUM" if worst_action == "ANONYMIZED" else "NONE"),
        })
    elif "sensitiveInformationPolicy" in assessment:
        # Policy ran and found nothing -- still a genuine PASS check.
        checks.append({
            "guardrail_type": "PII_CHECK", "result": "PASS", "confidence": 0.05, "severity_label": "NONE",
        })

    word_policy = assessment.get("wordPolicy", {})
    if word_policy:
        custom_words = word_policy.get("customWords", [])
        action = "BLOCKED" if any(w.get("action") == "BLOCKED" for w in custom_words) else "NONE"
        checks.append({
            "guardrail_type": "SENSITIVE_INFO_CHECK",
            "result": ACTION_TO_RESULT.get(action, "PASS"),
            "confidence": 0.8 if custom_words else 0.05,
            "severity_label": "HIGH" if action == "BLOCKED" else "NONE",
        })

    grounding_policy = assessment.get("contextualGroundingPolicy", {})
    for f in grounding_policy.get("filters", []):
        if f.get("type") != "GROUNDING":
            continue
        score = float(f.get("score", 1.0))
        threshold = float(f.get("threshold", 0.75))
        action = f.get("action", "NONE")
        checks.append({
            "guardrail_type": "GROUNDING_CHECK",
            "result": ACTION_TO_RESULT.get(action, "PASS" if score >= threshold else "WARN"),
            "confidence": round(score, 3),
            "severity_label": "NONE" if score >= threshold else "MEDIUM",
        })

    return checks


def adapt_bedrock_session(raw: Dict[str, Any]) -> pd.DataFrame:
    """Convert one Bedrock-style agent invocation session (as exported
    from InvokeAgent trace + ApplyGuardrail assessments) into a DataFrame
    matching this project's internal trace schema (TRACE_COLUMNS).
    """
    workflow_id = f"bedrock_{raw.get('agentId', 'unknown_agent')}"
    request_id = raw.get("sessionId", "unknown_session")

    rows: List[Dict[str, Any]] = []
    step_id = 0

    for event in sorted(raw.get("traceEvents", []), key=lambda e: e.get("eventOrder", 0)):
        event_type = event.get("eventType", "ORCHESTRATION")
        boundary = EVENT_TYPE_TO_BOUNDARY.get(event_type, "AGENT_TO_AGENT_INTERNAL")
        node_type = EVENT_TYPE_TO_NODE_TYPE.get(event_type, "agent")

        source_node = event.get("sourceNode", "User")
        target_node = event.get("targetNode", "FinalAnswer")

        input_text = event.get("inputText", "")
        output_text = event.get("outputText", input_text)
        input_hash = _hash_text(input_text)
        output_hash = _hash_text(output_text)
        text_changed = input_hash != output_hash

        risk_level = _infer_risk_level(event.get("riskTag"), source_node, target_node, event.get("modelId", ""))

        guardrail_trace = event.get("guardrailTrace")
        assessments = guardrail_trace.get("assessments", []) if guardrail_trace else []
        guardrail_id = guardrail_trace.get("guardrailId", "unspecified") if guardrail_trace else "unspecified"
        guardrail_version = guardrail_trace.get("guardrailVersion", "1") if guardrail_trace else "1"
        policy_id = f"{guardrail_id}:{guardrail_version}"

        latency_ms = float(event.get("latencyMs", 100.0))
        usage_units = float(event.get("usageUnits", 1.0))
        estimated_cost_usd = round(usage_units * COST_PER_GUARDRAIL_UNIT_USD, 6)

        checks: List[Dict[str, Any]] = []
        for assessment in assessments:
            checks.extend(_iter_assessment_checks(assessment))

        if not checks:
            # No guardrail ran on this hop at all -- still emit the hop as
            # a zero-guardrail row so the graph builder sees the edge, but
            # do not fabricate a guardrail_type that was never evaluated.
            continue

        for check in checks:
            step_id += 1
            rows.append({
                "workflow_id": workflow_id,
                "request_id": request_id,
                "step_id": step_id,
                "node_id": f"{workflow_id}_{target_node}",
                "source_node": source_node,
                "target_node": target_node,
                "node_type": node_type,
                "step_type": node_type,
                "agent_name": target_node,
                "guardrail_type": check["guardrail_type"],
                "policy_id": policy_id,
                "text_hash": output_hash,
                "input_text_hash": input_hash,
                "output_text_hash": output_hash,
                "text_changed": text_changed,
                "trust_boundary": boundary,
                "risk_level": risk_level,
                "edge_risk_level": risk_level,
                "guardrail_result": check["result"],
                "severity": check["severity_label"],
                "confidence": check["confidence"],
                "latency_ms": latency_ms,
                "estimated_cost_usd": estimated_cost_usd,
                "timestamp": event.get("timestamp", ""),
                "workflow_template": "bedrock_live_import",
                "behavior_pattern": "real_bedrock_trace",
            })

    df = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    return df


def adapt_bedrock_sessions(raw_sessions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Adapt multiple sessions (e.g. a batch CloudWatch export covering
    many InvokeAgent calls) into a single combined DataFrame."""
    frames = [adapt_bedrock_session(s) for s in raw_sessions]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_bedrock_trace_file(path: str) -> pd.DataFrame:
    """Load a JSON file containing either a single session object or a
    list of session objects, and adapt it into the internal schema."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return adapt_bedrock_sessions(data)
    return adapt_bedrock_session(data)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt real Bedrock-style trace/guardrail JSON into the internal graph schema. "
        "This does not call AWS -- it transforms a JSON file you already exported."
    )
    parser.add_argument("--input", required=True, help="Path to a Bedrock-style trace JSON file")
    parser.add_argument("--out", required=True, help="Path to write the adapted CSV")
    args = parser.parse_args()

    df = load_bedrock_trace_file(args.input)
    df.to_csv(args.out, index=False)
    print(f"Adapted {len(df)} guardrail-check rows from {args.input} -> {args.out}")
    if df.empty:
        print("Warning: zero rows produced. Check that traceEvents contain guardrailTrace.assessments.")


if __name__ == "__main__":
    _cli()
