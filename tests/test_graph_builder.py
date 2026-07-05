"""
test_graph_builder.py
----------------------
Tests proving graph_builder.py builds true directed graphs (nodes, edges,
paths, boundaries, attached guardrail checks) rather than flat row tables.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from synthetic_workflows import generate_synthetic_workflows
from graph_builder import build_all_workflow_graphs, build_workflow_graph, PROTECTED_BOUNDARIES


@pytest.fixture(scope="module")
def df():
    return generate_synthetic_workflows(n_workflows=40, seed=1)


@pytest.fixture(scope="module")
def graphs(df):
    return build_all_workflow_graphs(df)


def test_builds_graph_nodes_for_each_workflow(df, graphs):
    assert len(graphs) == df[["workflow_id", "request_id"]].drop_duplicates().shape[0]
    for wg in graphs.values():
        assert len(wg.nodes) > 0


def test_builds_graph_edges_correctly(df, graphs):
    for wg in graphs.values():
        assert len(wg.edges) > 0
        for (src, tgt), edge in wg.edges.items():
            assert edge.source_node == src
            assert edge.target_node == tgt
            assert wg.graph.has_edge(src, tgt)


def test_preserves_directed_order_from_user_to_final(graphs):
    """all_paths() now enumerates every terminal node, both real
    UserInput->FinalAnswer paths (path_type="final") and dead-end orphan
    branches (path_type="orphan_branch") that never reach FinalAnswer -- see
    the all_paths_with_types() fix in graph_builder.py. Every path must
    still start at User, and every path tagged "final" must still end at an
    actual final node.
    """
    found_linear = False
    for wg in graphs.values():
        if wg.user_node() == "User" and "FinalAnswer" in wg.graph.nodes:
            found_linear = True
            paths_with_types = wg.all_paths_with_types()
            assert len(paths_with_types) >= 1
            for p, path_type in paths_with_types:
                assert p[0] == "User"
                if path_type == "final":
                    assert p[-1] in wg.final_nodes()
                else:
                    assert path_type == "orphan_branch"
                    assert p[-1] not in wg.final_nodes()
    assert found_linear


def test_orphan_branch_paths_never_reach_final_and_are_never_dropped(graphs):
    """Regression test for the correctness bug: a workflow can contain BOTH
    a real UserInput->FinalAnswer path and a separate dead-end branch (e.g.
    Planner->Retrieval->Summarizer with no onward edge) in the same graph.
    Every edge -- including ones on the dead-end branch -- must be covered
    by at least one enumerated path.
    """
    found_mixed = False
    for wg in graphs.values():
        paths_with_types = wg.all_paths_with_types()
        path_types = {t for _, t in paths_with_types}
        if "orphan_branch" in path_types and "final" in path_types:
            found_mixed = True
            covered_edges = set()
            for p, _t in paths_with_types:
                for i in range(len(p) - 1):
                    covered_edges.add((p[i], p[i + 1]))
            assert covered_edges == set(wg.edges.keys()), (
                f"Not every edge in {wg.workflow_id}/{wg.request_id} is covered by an "
                f"enumerated path: missing {set(wg.edges.keys()) - covered_edges}"
            )
    assert found_mixed, "Expected at least one workflow with both a final path and an orphan branch."


def test_finds_all_paths_from_user_to_final(graphs):
    multi_path_found = any(len(wg.all_paths()) > 1 for wg in graphs.values())
    assert multi_path_found


def test_attaches_guardrail_checks_to_graph_nodes_and_edges(graphs):
    any_checks = False
    for wg in graphs.values():
        for edge in wg.edges.values():
            if len(edge.guardrail_checks) > 0:
                any_checks = True
                for check in edge.guardrail_checks:
                    assert "guardrail_type" in check
        for node in wg.nodes.values():
            if len(node.guardrail_checks) > 0:
                any_checks = True
    assert any_checks


def test_detects_protected_trust_boundaries(graphs):
    protected_found = False
    optimizable_found = False
    for wg in graphs.values():
        for edge in wg.edges.values():
            if edge.trust_boundary in PROTECTED_BOUNDARIES:
                protected_found = True
            if edge.trust_boundary == "AGENT_TO_AGENT_INTERNAL":
                optimizable_found = True
    assert protected_found
    assert optimizable_found


def test_handles_branching_workflow(df):
    branch_rows = df[df["target_node"] == "ComplianceAgent"]
    assert len(branch_rows) > 0
    wf_id = branch_rows.iloc[0]["workflow_id"]
    req_id = branch_rows.iloc[0]["request_id"]
    wg = build_workflow_graph(df, wf_id, req_id)
    successors = list(wg.graph.successors("PlannerAgent"))
    assert len(successors) >= 2
    paths = wg.all_paths()
    assert len(paths) >= 2
