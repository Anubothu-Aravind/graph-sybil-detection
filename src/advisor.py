"""Automatic dataset analyzer and heuristic advisor for GraphShield.

Computes structural characteristics and recommends sensible initial model hyperparameters.
"""

from typing import Dict, Any, Optional
import numpy as np
import networkx as nx

from src.utils import get_device_info


def analyze_and_recommend(
    G: nx.Graph,
    known_labels: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Inspect graph structure and generate data-driven algorithm recommendations.

    Args:
        G: NetworkX graph.
        known_labels: Dict mapping node_id -> integer label (-1 for unlabeled).

    Returns:
        Structured advisor report with size, connectivity, label coverage, and smart defaults.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {
            "status": "empty",
            "message": "Graph contains no nodes.",
            "recommendations": {}
        }

    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    isolated_nodes = sum(1 for d in degrees if d == 0)

    # Graph density
    possible_edges = (num_nodes * (num_nodes - 1)) / 2 if not G.is_directed() else (num_nodes * (num_nodes - 1))
    density = float(num_edges / possible_edges) if possible_edges > 0 else 0.0

    # Connected components
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    num_components = len(components)
    largest_comp_size = max(len(c) for c in components) if components else 0
    largest_comp_ratio = float(largest_comp_size / num_nodes) if num_nodes > 0 else 0.0

    # Label statistics
    if known_labels is None:
        known_labels = {n: G.nodes[n].get("label", -1) for n in G.nodes()}

    labeled_count = sum(1 for lbl in known_labels.values() if lbl != -1)
    unlabeled_count = num_nodes - labeled_count
    label_coverage_ratio = float(labeled_count / num_nodes) if num_nodes > 0 else 0.0

    # Class breakdown
    unique_classes = set(lbl for lbl in known_labels.values() if lbl != -1)
    class_counts = {c: sum(1 for lbl in known_labels.values() if lbl == c) for c in unique_classes}

    # Size Categorization
    if num_nodes < 500:
        size_category = "Small (<500 nodes)"
        size_tag = "small"
    elif num_nodes <= 5000:
        size_category = "Medium (500-5,000 nodes)"
        size_tag = "medium"
    else:
        size_category = "Large (>5,000 nodes)"
        size_tag = "large"

    # Connectivity Categorization
    if num_components == 1:
        connectivity_status = "Fully connected"
    elif largest_comp_ratio >= 0.85:
        connectivity_status = f"Mostly connected ({largest_comp_ratio * 100:.1f}% in main component)"
    else:
        connectivity_status = f"Fragmented ({num_components} disconnected components)"

    # Label Coverage Categorization
    if label_coverage_ratio < 0.10:
        coverage_status = f"Low ({label_coverage_ratio * 100:.1f}%)"
    elif label_coverage_ratio <= 0.30:
        coverage_status = f"Moderate ({label_coverage_ratio * 100:.1f}%)"
    else:
        coverage_status = f"High ({label_coverage_ratio * 100:.1f}%)"

    # Hardware-informed recommendation
    dev_info = get_device_info()
    if dev_info["mps_available"] and num_nodes >= 500:
        recommended_algo = "GraphShield (Apple MPS / PyTorch Accelerated)"
    else:
        recommended_algo = "GraphShield (Custom LP - NumPy/SciPy)"

    # Heuristic Hyperparameters
    # For denser networks, alpha 0.85 is stable; for sparser networks, alpha 0.90 ensures diffusion penetrates deeper
    suggested_alpha = 0.85 if avg_degree >= 8.0 else 0.90
    suggested_max_iter = 100 if size_tag != "large" else 150
    suggested_tol = 0.001
    suggested_threshold = 0.50
    suggested_clamping = True
    suggested_normalization = "random_walk"

    # Structural Warnings & Contextual Explanations
    advisory_notes = []
    if isolated_nodes > 0:
        advisory_notes.append(
            f"Graph has {isolated_nodes} isolated node(s) with 0 connections. "
            "Isolated nodes cannot receive propagated signals from neighbors and will retain prior baseline beliefs."
        )

    if num_components > 1:
        advisory_notes.append(
            f"Graph is divided into {num_components} separate components. "
            "Diffusion cannot cross partition boundaries. Nodes in components without seed labels will remain neutral."
        )

    if label_coverage_ratio < 0.05 and labeled_count > 0:
        advisory_notes.append(
            f"Known seed coverage is under 5% ({labeled_count} nodes). "
            "Label propagation may require additional iterations or a higher alpha to diffuse across the full network."
        )

    if len(unique_classes) < 2 and labeled_count > 0:
        advisory_notes.append(
            "Known seeds contain only one active class. At least two classes (e.g. Normal and Sybil) "
            "are recommended for competitive label propagation."
        )

    return {
        "status": "complete",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": round(avg_degree, 2),
        "density": round(density, 6),
        "num_components": num_components,
        "isolated_nodes": isolated_nodes,
        "largest_comp_size": largest_comp_size,
        "largest_comp_ratio": round(largest_comp_ratio, 4),
        "labeled_count": labeled_count,
        "unlabeled_count": unlabeled_count,
        "label_coverage_ratio": round(label_coverage_ratio, 4),
        "class_counts": class_counts,
        "size_category": size_category,
        "size_tag": size_tag,
        "connectivity_status": connectivity_status,
        "coverage_status": coverage_status,
        "advisory_notes": advisory_notes,
        "recommendations": {
            "algorithm": recommended_algo,
            "alpha": suggested_alpha,
            "max_iter": suggested_max_iter,
            "tol": suggested_tol,
            "threshold": suggested_threshold,
            "clamping": suggested_clamping,
            "normalization": suggested_normalization,
            "device": dev_info["recommended_device"]
        }
    }
