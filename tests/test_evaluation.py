"""Unit tests for src/evaluation.py."""

import pytest
import networkx as nx
from src.evaluation import (
    split_train_test_labels,
    compute_classification_metrics,
    run_benchmark_comparison
)


def test_split_train_test_labels_no_leakage():
    ground_truth = {f"N_{i}": 0 for i in range(20)}
    ground_truth.update({f"S_{j}": 1 for j in range(10)})

    train_labels, test_labels = split_train_test_labels(ground_truth, train_ratio=0.30, seed=42)

    # All nodes must exist in train_labels
    assert len(train_labels) == 30

    train_nodes = {n for n, l in train_labels.items() if l != -1}
    test_nodes = set(test_labels.keys())

    # Disjoint sets: no overlap between training seeds and held-out test labels
    assert len(train_nodes.intersection(test_nodes)) == 0

    # In train_labels, every held-out test node must be marked as unlabeled (-1)
    for t_node in test_nodes:
        assert train_labels[t_node] == -1, f"Test node {t_node} was not masked in train_labels!"
        # Ground truth matches
        assert test_labels[t_node] == ground_truth[t_node]


def test_compute_classification_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_run_benchmark_comparison():
    # Build simple graph
    G = nx.cycle_graph(10)
    G = nx.relabel_nodes(G, {i: f"node_{i}" for i in range(10)})
    gt = {f"node_{i}": (0 if i < 5 else 1) for i in range(10)}

    custom_m, baseline_m, df_comp = run_benchmark_comparison(
        G, gt, train_ratio=0.4, seed=42
    )

    assert "accuracy" in custom_m
    assert "accuracy" in baseline_m
    assert len(df_comp) == 6
    assert "Metric" in df_comp.columns
