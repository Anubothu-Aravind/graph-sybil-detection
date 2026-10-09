"""Unit tests for src/label_propagation.py."""

import pytest
import networkx as nx
from src.label_propagation import GraphShieldLabelPropagation, BaselineLabelSpreading


def create_barbell_test_graph():
    """Create a barbell-like graph with 2 cliques connected by a bridge.

    Clique 1 (Normal): N1, N2, N3, N4
    Clique 2 (Sybil):  S1, S2, S3, S4
    Bridge: N4 -- S1
    """
    G = nx.Graph()
    # Clique 1
    norm_nodes = ["N1", "N2", "N3", "N4"]
    for i in range(len(norm_nodes)):
        for j in range(i + 1, len(norm_nodes)):
            G.add_edge(norm_nodes[i], norm_nodes[j])

    # Clique 2
    sybil_nodes = ["S1", "S2", "S3", "S4"]
    for i in range(len(sybil_nodes)):
        for j in range(i + 1, len(sybil_nodes)):
            G.add_edge(sybil_nodes[i], sybil_nodes[j])

    # Bridge
    G.add_edge("N4", "S1")

    # Known seeds: N1 is Normal (0), S4 is Sybil (1). All others unlabeled (-1).
    known_labels = {
        "N1": 0, "N2": -1, "N3": -1, "N4": -1,
        "S1": -1, "S2": -1, "S3": -1, "S4": 1
    }
    return G, known_labels


def test_label_propagation_barbell_graph():
    G, known_labels = create_barbell_test_graph()

    model = GraphShieldLabelPropagation(
        alpha=0.85,
        max_iter=100,
        tol=1e-5,
        clamping=True,
        normalization="random_walk",
        threshold=0.50
    )

    results = model.fit_predict(G, known_labels)
    preds = results["predictions"]
    confs = results["confidence_scores"]

    # In Clique 1, all nodes should be predicted as Normal (0)
    for n in ["N1", "N2", "N3", "N4"]:
        assert preds[n] == 0, f"Expected Normal for {n}, got {preds[n]}"
        assert confs[n] < 0.5, f"Expected confidence < 0.5 for {n}, got {confs[n]}"

    # In Clique 2, all nodes should be predicted as Sybil (1)
    for s in ["S1", "S2", "S3", "S4"]:
        assert preds[s] == 1, f"Expected Sybil for {s}, got {preds[s]}"
        assert confs[s] > 0.5, f"Expected confidence > 0.5 for {s}, got {confs[s]}"

    assert results["converged"] is True


def test_clamping_preservation():
    G, known_labels = create_barbell_test_graph()

    model = GraphShieldLabelPropagation(clamping=True)
    results = model.fit_predict(G, known_labels)

    # Clamped nodes must retain exact original labels
    assert results["predictions"]["N1"] == 0
    assert results["confidence_scores"]["N1"] == 0.0
    assert results["predictions"]["S4"] == 1
    assert results["confidence_scores"]["S4"] == 1.0


def test_normalization_modes():
    G, known_labels = create_barbell_test_graph()

    for norm in ["random_walk", "normalized_laplacian"]:
        model = GraphShieldLabelPropagation(normalization=norm)
        res = model.fit_predict(G, known_labels)
        assert len(res["predictions"]) == 8
        for n, prob in res["probabilities"].items():
            assert pytest.approx(sum(prob), 0.01) == 1.0


def test_baseline_label_spreading():
    G, known_labels = create_barbell_test_graph()

    baseline = BaselineLabelSpreading(alpha=0.85)
    res = baseline.fit_predict(G, known_labels)

    assert "predictions" in res
    assert res["predictions"]["N1"] == 0
    assert res["predictions"]["S4"] == 1


def test_label_propagation_torch_backend():
    import torch
    from src.utils import get_device_info

    info = get_device_info()
    device = "mps" if info["mps_available"] else "cpu"

    G, known_labels = create_barbell_test_graph()
    model = GraphShieldLabelPropagation(use_torch=True, device=device)
    res = model.fit_predict(G, known_labels)

    assert len(res["predictions"]) == 8
    assert res["predictions"]["N1"] == 0
    assert res["predictions"]["S4"] == 1
    assert "backend" in res
