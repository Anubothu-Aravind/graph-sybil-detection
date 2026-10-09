"""Unit tests for src/data_loader.py."""

import io
import pytest
import pandas as pd
import networkx as nx

from src.data_loader import (
    load_and_validate_edges,
    load_and_validate_labels,
    compute_graph_statistics,
    ValidationError
)


def test_load_and_validate_edges_valid():
    csv_data = "source,target\n1,2\n2,3\n3,1\n"
    df, report = load_and_validate_edges(io.StringIO(csv_data))
    assert len(df) == 3
    assert report["clean_edge_count"] == 3
    assert report["self_loops_removed"] == 0
    assert report["duplicates_removed"] == 0


def test_load_and_validate_edges_self_loops_and_duplicates():
    # Includes self loop (2,2) and duplicate (1,2) / (2,1) in undirected graph
    csv_data = "source,target\n1,2\n2,1\n2,2\n2,3\n"
    df, report = load_and_validate_edges(io.StringIO(csv_data), directed=False)
    assert report["self_loops_removed"] == 1
    assert report["duplicates_removed"] == 1
    assert len(df) == 2


def test_load_and_validate_edges_missing_columns():
    csv_data = "colA,colB\n1,2\n"
    with pytest.raises(ValidationError):
        load_and_validate_edges(io.StringIO(csv_data))


def test_load_and_validate_labels_valid():
    csv_data = "node_id,label\n1,0\n2,1\n3,\n4,-1\n5,invalid\n"
    df, report = load_and_validate_labels(io.StringIO(csv_data))
    assert len(df) == 5
    assert report["normal_count"] == 1
    assert report["sybil_count"] == 1
    # 3, 4, 5 should all be treated as unlabeled (-1)
    assert report["unlabeled_count"] == 3
    assert report["invalid_label_values"] == 1


def test_load_and_validate_labels_missing_columns():
    csv_data = "id,status\n1,0\n"
    with pytest.raises(ValidationError):
        load_and_validate_labels(io.StringIO(csv_data))


def test_compute_graph_statistics():
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C")])
    G.add_node("D")  # Isolated node
    labels = {"A": 0, "B": 0, "C": 1, "D": -1}

    stats = compute_graph_statistics(G, labels)
    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 2
    assert stats["isolated_nodes"] == 1
    assert stats["num_components"] == 2
    assert stats["normal_count"] == 2
    assert stats["sybil_count"] == 1
    assert stats["unlabeled_count"] == 1
    assert stats["sybil_ratio_among_labeled"] == pytest.approx(1 / 3, 0.01)
