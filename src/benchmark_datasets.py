"""Benchmark dataset loaders for academic graph networks (Cora, PubMed) via PyTorch Geometric."""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


# Topic name mappings for transparency
CORA_TOPICS = {
    0: "Rule Learning",
    1: "Neural Networks",
    2: "Probabilistic Methods",
    3: "Genetic Algorithms",
    4: "Theory",
    5: "Reinforcement Learning",
    6: "Case-Based Reasoning"
}

PUBMED_TOPICS = {
    0: "Diabetes Mellitus, Experimental",
    1: "Diabetes Mellitus Type 1",
    2: "Diabetes Mellitus Type 2"
}


def load_cora_dataset(root: str = "data/planetoid") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and process the Cora citation benchmark network using PyTorch Geometric.

    Returns:
        edges_df: DataFrame of citation edges ['source', 'target']
        labels_df: DataFrame of semi-supervised labels ['node_id', 'label'] (-1 for unlabeled)
        ground_truth_df: DataFrame of all true paper topics ['node_id', 'ground_truth']
        metadata: Detailed dataset metadata and academic disclosure
    """
    return _load_planetoid("Cora", root, CORA_TOPICS)


def load_pubmed_dataset(root: str = "data/planetoid") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and process the PubMed citation benchmark network using PyTorch Geometric.

    Returns:
        edges_df: DataFrame of citation edges ['source', 'target']
        labels_df: DataFrame of semi-supervised labels ['node_id', 'label'] (-1 for unlabeled)
        ground_truth_df: DataFrame of all true paper topics ['node_id', 'ground_truth']
        metadata: Detailed dataset metadata and academic disclosure
    """
    return _load_planetoid("PubMed", root, PUBMED_TOPICS)


def _load_planetoid(
    name: str,
    root: str,
    topic_mapping: Dict[int, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Helper to load and format PyTorch Geometric Planetoid datasets."""
    os.makedirs(root, exist_ok=True)

    try:
        from torch_geometric.datasets import Planetoid
    except ImportError as e:
        raise RuntimeError(f"PyTorch Geometric is required to load {name}. Install via pip install torch_geometric: {e}")

    dataset = Planetoid(root=root, name=name)
    data = dataset[0]

    num_nodes = int(data.num_nodes)
    edges = data.edge_index.numpy()
    src = edges[0]
    tgt = edges[1]

    # Drop self-loops and canonicalize undirected edges
    mask = src != tgt
    src_clean = src[mask]
    tgt_clean = tgt[mask]

    # Canonicalize for undirected representation
    u = np.where(src_clean <= tgt_clean, src_clean, tgt_clean)
    v = np.where(src_clean <= tgt_clean, tgt_clean, src_clean)

    raw_edges_df = pd.DataFrame({"source": [str(x) for x in u], "target": [str(y) for y in v]})
    edges_df = raw_edges_df.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)

    # Node IDs and labels
    node_ids = [str(i) for i in range(num_nodes)]
    y = data.y.numpy()

    gt_df = pd.DataFrame({
        "node_id": node_ids,
        "ground_truth": [int(val) for val in y]
    })

    # Standard Planetoid train_mask for semi-supervised training seeds
    if hasattr(data, "train_mask") and data.train_mask is not None:
        train_mask = data.train_mask.numpy()
    else:
        # Fallback to ~10% training seeds if train_mask not available
        rng = np.random.default_rng(42)
        train_mask = rng.random(num_nodes) < 0.10

    labels = []
    for i in range(num_nodes):
        if train_mask[i]:
            labels.append(int(y[i]))
        else:
            labels.append(-1)

    labels_df = pd.DataFrame({
        "node_id": node_ids,
        "label": labels
    })

    num_classes = int(dataset.num_classes)
    num_labeled = int(np.sum(train_mask))

    metadata = {
        "dataset_name": f"{name} Citation Network",
        "dataset_key": name.lower(),
        "is_synthetic": False,
        "is_citation_benchmark": True,
        "num_total_nodes": num_nodes,
        "num_edges": len(edges_df),
        "num_classes": num_classes,
        "num_labeled_nodes": num_labeled,
        "labeled_ratio": round(num_labeled / num_nodes, 4),
        "topic_mapping": topic_mapping,
        "notice": (
            f"Academic citation network. Nodes represent scientific publications; "
            f"edges represent citation relationships. Node labels represent paper topics "
            f"(not Sybil accounts). Used for graph scale and semi-supervised benchmark evaluation."
        )
    }

    return edges_df, labels_df, gt_df, metadata
