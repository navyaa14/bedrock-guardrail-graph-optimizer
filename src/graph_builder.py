"""
graph_builder.py
------------------
Builds a true directed-graph representation of Bedrock-style multi-agent
workflows from the flat guardrail-check trace DataFrame.

Each (workflow_id, request_id) pair becomes a `WorkflowGraph`: a
networkx.DiGraph where nodes are UserInput / agents / tools / FinalAnswer,
and edges are agent-to-agent (or agent-to-tool, etc.) handoffs. Guardrail
checks are attached to the edge they were executed on (edges carry the
trust boundary they cross), which is what graph_features.py,
graph_optimizer.py, and graph_metrics.py all operate on.

networkx is used for real graph operations (all simple paths, topological
structure, etc.) rather than re-implementing graph traversal by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import networkx as nx

PROTECTED_BOUNDARIES = {
    "USER_TO_AGENT",
    "AGENT_TO_TOOL",
    "TOOL_TO_AGENT",
    "AGENT_TO_FINAL_RESPONSE",
}
OPTIMIZABLE_BOUNDARIES = {"AGENT_TO_AGENT_INTERNAL"}


@dataclass
class GraphNode:
    node_id: str
    workflow_id: str
    request_id: str
    node_name: str
    node_type: str  # user, agent, tool, final
    agent_name: str
    risk_level: str
    input_text_hash: str
    output_text_hash: str
    guardrail_checks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GraphEdge:
    edge_id: str
    workflow_id: str
    request_id: str
    source_node: str
    target_node: str
    trust_boundary: str
    text_changed_on_edge: bool
    edge_risk_level: str
    guardrail_checks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowGraph:
    """Graph representation for a single (workflow_id, request_id) pair."""

    workflow_id: str
    request_id: str
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], GraphEdge] = field(default_factory=dict)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def user_node(self) -> Optional[str]:
        for n in self.graph.nodes:
            if self.graph.nodes[n].get("node_type") == "user":
                return n
        return None

    def final_nodes(self) -> List[str]:
        return [n for n in self.graph.nodes if self.graph.nodes[n].get("node_type") == "final"]

    def all_paths_with_types(self) -> List[Tuple[List[str], str]]:
        """Enumerate every path needed to guarantee full edge coverage.

        Returns a list of (path, path_type) tuples where path_type is one of:
          - "final":          a real UserInput -> FinalAnswer path.
          - "orphan_branch":  a dead-end branch that never reaches a final
                               node (e.g. Planner -> Retrieval -> Summarizer
                               with no onward edge), enumerated so that its
                               guardrail checks are still routed through the
                               decision engine instead of being silently
                               dropped just because `all_paths()` used to
                               only enumerate root->final paths.

        Guarantee: every edge in the graph is covered by at least one
        returned path, regardless of whether the graph has a reachable
        final node at all (the old fallback-to-leaves behavior only kicked
        in when NO final node existed anywhere -- that was the bug: a
        workflow with both a real final path AND a separate dead-end branch
        would silently drop the dead branch's checks).
        """
        start = self.user_node()
        if start is None:
            return []

        results: List[Tuple[List[str], str]] = []
        covered_edges: set = set()

        finals = self.final_nodes()
        for end in finals:
            if end == start:
                continue
            for p in nx.all_simple_paths(self.graph, source=start, target=end):
                results.append((p, "final"))
                for i in range(len(p) - 1):
                    covered_edges.add((p[i], p[i + 1]))

        all_edges = set(self.graph.edges())
        remaining = all_edges - covered_edges
        if remaining:
            # Any node with out-degree 0 that isn't already a covered final
            # node is a candidate dead-end. Enumerate root->leaf paths for
            # each such leaf and keep the ones that actually cover at least
            # one still-uncovered edge (avoids duplicate/empty paths).
            leaves = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0 and n != start]
            for leaf in leaves:
                if leaf in finals:
                    continue
                for p in nx.all_simple_paths(self.graph, source=start, target=leaf):
                    edge_set = {(p[i], p[i + 1]) for i in range(len(p) - 1)}
                    if edge_set & remaining:
                        results.append((p, "orphan_branch"))
                        covered_edges |= edge_set
            remaining = all_edges - covered_edges

        return results

    def all_paths(self) -> List[List[str]]:
        """Enumerate all paths needed for full edge coverage (final paths
        plus any dead-end orphan branches). See `all_paths_with_types()`
        for the path_type breakdown; this is the flat list of node-lists
        for callers that don't need the type tag.
        """
        return [p for p, _ in self.all_paths_with_types()]

    def has_final_boundary_check(self) -> bool:
        """Whether any edge with trust_boundary AGENT_TO_FINAL_RESPONSE has >=1 check."""
        for edge in self.edges.values():
            if edge.trust_boundary == "AGENT_TO_FINAL_RESPONSE" and len(edge.guardrail_checks) > 0:
                return True
        return False

    def has_final_boundary_edge(self) -> bool:
        return any(e.trust_boundary == "AGENT_TO_FINAL_RESPONSE" for e in self.edges.values())


def build_workflow_graph(df: pd.DataFrame, workflow_id: str, request_id: str) -> WorkflowGraph:
    """Build a `WorkflowGraph` for one workflow/request from the trace DataFrame."""
    subset = df[(df["workflow_id"] == workflow_id) & (df["request_id"] == request_id)].sort_values("step_id")

    nodes: Dict[str, GraphNode] = {}
    edges: Dict[Tuple[str, str], GraphEdge] = {}
    nxg = nx.DiGraph()

    for _, row in subset.iterrows():
        src, tgt = row["source_node"], row["target_node"]

        if src not in nodes:
            src_type = "user" if src == "User" else "agent"
            nodes[src] = GraphNode(
                node_id=f"{request_id}_{src}",
                workflow_id=workflow_id,
                request_id=request_id,
                node_name=src,
                node_type=src_type,
                agent_name=src,
                risk_level=row.get("risk_level", "low"),
                input_text_hash=row.get("input_text_hash", ""),
                output_text_hash=row.get("input_text_hash", ""),
            )
            nxg.add_node(src, node_type=src_type)

        if tgt not in nodes:
            nodes[tgt] = GraphNode(
                node_id=row.get("node_id", f"{request_id}_{tgt}"),
                workflow_id=workflow_id,
                request_id=request_id,
                node_name=tgt,
                node_type=row.get("node_type", "agent"),
                agent_name=row.get("agent_name", tgt),
                risk_level=row.get("risk_level", "low"),
                input_text_hash=row.get("input_text_hash", ""),
                output_text_hash=row.get("output_text_hash", ""),
            )
            nxg.add_node(tgt, node_type=row.get("node_type", "agent"))

        nodes[tgt].guardrail_checks.append({str(k): v for k, v in row.to_dict().items()})

        edge_key = (src, tgt)
        if edge_key not in edges:
            edges[edge_key] = GraphEdge(
                edge_id=f"{request_id}_{src}_{tgt}",
                workflow_id=workflow_id,
                request_id=request_id,
                source_node=src,
                target_node=tgt,
                trust_boundary=row["trust_boundary"],
                text_changed_on_edge=bool(row.get("text_changed", False)),
                edge_risk_level=row.get("edge_risk_level", row.get("risk_level", "low")),
            )
            nxg.add_edge(
                src,
                tgt,
                trust_boundary=row["trust_boundary"],
                protected=row["trust_boundary"] in PROTECTED_BOUNDARIES,
            )
        edges[edge_key].guardrail_checks.append({str(k): v for k, v in row.to_dict().items()})
        # If any check on this edge saw a text change, mark the edge as such.
        if bool(row.get("text_changed", False)):
            edges[edge_key].text_changed_on_edge = True

    return WorkflowGraph(
        workflow_id=workflow_id,
        request_id=request_id,
        nodes=nodes,
        edges=edges,
        graph=nxg,
    )


def build_all_workflow_graphs(df: pd.DataFrame) -> Dict[Tuple[str, str], WorkflowGraph]:
    """Build a `WorkflowGraph` for every (workflow_id, request_id) pair in df."""
    graphs: Dict[Tuple[str, str], WorkflowGraph] = {}
    pairs = df[["workflow_id", "request_id"]].drop_duplicates()
    for _, pair_row in pairs.iterrows():
        wf_id, req_id = pair_row["workflow_id"], pair_row["request_id"]
        graphs[(wf_id, req_id)] = build_workflow_graph(df, wf_id, req_id)
    return graphs


def graphs_to_node_table(graphs: Dict[Tuple[str, str], WorkflowGraph]) -> pd.DataFrame:
    """Flatten all WorkflowGraph nodes into a DataFrame -> outputs/graph_nodes.csv."""
    rows = []
    for wg in graphs.values():
        for node in wg.nodes.values():
            rows.append(
                {
                    "node_id": node.node_id,
                    "workflow_id": node.workflow_id,
                    "request_id": node.request_id,
                    "node_name": node.node_name,
                    "node_type": node.node_type,
                    "agent_name": node.agent_name,
                    "risk_level": node.risk_level,
                    "input_text_hash": node.input_text_hash,
                    "output_text_hash": node.output_text_hash,
                    "guardrail_check_count": len(node.guardrail_checks),
                }
            )
    return pd.DataFrame(rows)


def graphs_to_edge_table(graphs: Dict[Tuple[str, str], WorkflowGraph]) -> pd.DataFrame:
    """Flatten all WorkflowGraph edges into a DataFrame -> outputs/graph_edges.csv."""
    rows = []
    for wg in graphs.values():
        for edge in wg.edges.values():
            rows.append(
                {
                    "edge_id": edge.edge_id,
                    "workflow_id": edge.workflow_id,
                    "request_id": edge.request_id,
                    "source_node": edge.source_node,
                    "target_node": edge.target_node,
                    "trust_boundary": edge.trust_boundary,
                    "protected_boundary": edge.trust_boundary in PROTECTED_BOUNDARIES,
                    "text_changed_on_edge": edge.text_changed_on_edge,
                    "edge_risk_level": edge.edge_risk_level,
                    "guardrail_check_count": len(edge.guardrail_checks),
                }
            )
    return pd.DataFrame(rows)


def save_graph_tables(graphs: Dict[Tuple[str, str], WorkflowGraph], nodes_path: str, edges_path: str) -> None:
    graphs_to_node_table(graphs).to_csv(nodes_path, index=False)
    graphs_to_edge_table(graphs).to_csv(edges_path, index=False)
