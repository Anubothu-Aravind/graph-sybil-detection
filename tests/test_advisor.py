"""Unit tests for src/advisor.py."""

import pytest
import networkx as nx
from src.advisor import analyze_and_recommend


def test_analyze_and_recommend_connected():
    G = nx.cycle_graph(20)
    known_labels = {i: (0 if i < 3 else (1 if i == 5 else -1)) for i in range(20)}

    report = analyze_and_recommend(G, known_labels)

    assert report["status"] == "complete"
    assert report["num_nodes"] == 20
    assert report["num_edges"] == 20
    assert report["num_components"] == 1
    assert "Fully connected" in report["connectivity_status"]
    assert report["size_tag"] == "small"
    assert "recommendations" in report
    assert report["recommendations"]["alpha"] in [0.85, 0.90]
    assert report["recommendations"]["threshold"] == 0.50


def test_analyze_and_recommend_fragmented():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(3, 4)
    G.add_node(5)  # Isolated node
    known_labels = {1: 0, 2: -1, 3: 1, 4: -1, 5: -1}

    report = analyze_and_recommend(G, known_labels)

    assert report["num_nodes"] == 5
    assert report["isolated_nodes"] == 1
    assert report["num_components"] == 3
    assert "Fragmented" in report["connectivity_status"]
    assert len(report["advisory_notes"]) > 0
