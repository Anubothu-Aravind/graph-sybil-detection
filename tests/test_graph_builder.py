"""Unit tests for src/graph_builder.py."""

import pytest
import pandas as pd
import networkx as nx

from src.graph_builder import build_networkx_graph, graph_to_sparse_adjacency


def test_build_networkx_graph_basic():
    edges_df = pd.DataFrame([
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
    ])
    labels_df = pd.DataFrame([
        {"node_id": "1", "label": 0},
        {"node_id": "3", "label": 1},
        {"node_id": "4", "label": -1},  # Isolated node in labels
    ])

    G, info = build_networkx_graph(edges_df, labels_df, directed=False)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 2
    assert info["isolated_nodes_count"] == 1
    assert "4" in info["isolated_nodes_list"]

    # Verify node features
    assert G.nodes["1"]["label"] == 0
    assert G.nodes["3"]["label"] == 1
    assert G.nodes["2"]["label"] == -1
    assert G.nodes["2"]["degree"] == 2
    # Node 2 connects to node 3 (known sybil), so suspicious neighbors = 1
    assert G.nodes["2"]["known_suspicious_neighbors"] == 1


def test_graph_to_sparse_adjacency():
    G = nx.Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")

    A, nodes, node_to_idx = graph_to_sparse_adjacency(G)
    assert A.shape == (3, 3)
    assert len(nodes) == 3
    # A connects to B
    i, j = node_to_idx["A"], node_to_idx["B"]
    assert A[i, j] == 1.0
    assert A[j, i] == 1.0
