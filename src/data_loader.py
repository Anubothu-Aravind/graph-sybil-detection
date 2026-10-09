"""Data loading, validation, and graph dataset statistics calculation for GraphShield."""

import io
from typing import Union, Tuple, Dict, Any, Optional, Set
import numpy as np
import pandas as pd
import networkx as nx


class ValidationError(Exception):
    """Raised when uploaded datasets fail schema or content validation."""
    pass


def load_and_validate_edges(
    source: Union[str, io.IOBase, pd.DataFrame],
    directed: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate an edge list CSV or DataFrame.

    Expected columns: 'source', 'target'.

    Returns:
        clean_df: Cleaned DataFrame with columns ['source', 'target'].
        report: Dictionary summarizing validation findings (duplicate edges, self-loops, etc.)
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        try:
            df = pd.read_csv(source)
        except Exception as e:
            raise ValidationError(f"Failed to parse edges CSV: {e}")

    # Column verification (case-insensitive strip)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "source" not in df.columns or "target" not in df.columns:
        raise ValidationError(
            f"Edges dataset must contain 'source' and 'target' columns. Found columns: {list(df.columns)}"
        )

    raw_count = len(df)
    if raw_count == 0:
        raise ValidationError("Edges dataset is empty.")

    # Drop null or blank nodes
    df = df.dropna(subset=["source", "target"])
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"] != "") & (df["target"] != "")]

    null_count = raw_count - len(df)

    # Detect self-loops
    self_loops_mask = df["source"] == df["target"]
    self_loop_count = int(self_loops_mask.sum())
    clean_df = df[~self_loops_mask].copy()

    # Deduplicate edges
    if not directed:
        # Standardize pair ordering so (u, v) and (v, u) are recognized as identical
        u = clean_df["source"].values
        v = clean_df["target"].values
        canonical_src = np.where(u <= v, u, v)
        canonical_tgt = np.where(u <= v, v, u)
        clean_df["_u"] = canonical_src
        clean_df["_v"] = canonical_tgt
        pre_dedup = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["_u", "_v"]).copy()
        dup_count = pre_dedup - len(clean_df)
        clean_df = clean_df[["_u", "_v"]].rename(columns={"_u": "source", "_v": "target"})
    else:
        pre_dedup = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["source", "target"]).copy()
        dup_count = pre_dedup - len(clean_df)
        clean_df = clean_df[["source", "target"]]

    clean_df.reset_index(drop=True, inplace=True)

    report = {
        "raw_edge_count": raw_count,
        "clean_edge_count": len(clean_df),
        "null_edges_dropped": null_count,
        "self_loops_removed": self_loop_count,
        "duplicates_removed": dup_count,
        "is_directed": directed,
    }

    return clean_df, report


def load_and_validate_labels(
    source: Union[str, io.IOBase, pd.DataFrame],
    known_nodes: Optional[Set[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate node labels CSV or DataFrame.

    Expected columns: 'node_id', 'label'.
    Labels: 0 (Normal), 1 (Sybil), or NaN/-1 for unlabeled.

    Returns:
        clean_df: Cleaned DataFrame with columns ['node_id', 'label'] where label is 0, 1, or -1.
        report: Dictionary summarizing label distribution and validation findings.
    """
    if source is None:
        return pd.DataFrame(columns=["node_id", "label"]), {"total_labels": 0}

    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        try:
            df = pd.read_csv(source)
        except Exception as e:
            raise ValidationError(f"Failed to parse labels CSV: {e}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "node_id" not in df.columns or "label" not in df.columns:
        raise ValidationError(
            f"Labels dataset must contain 'node_id' and 'label' columns. Found columns: {list(df.columns)}"
        )

    raw_count = len(df)
    df["node_id"] = df["node_id"].astype(str).str.strip()
    df = df[df["node_id"] != ""].copy()

    # Deduplicate node_id (keep first)
    dedup_labels = len(df) - len(df["node_id"].unique())
    df = df.drop_duplicates(subset=["node_id"], keep="first").copy()

    # Process labels: 0=Normal, 1=Sybil, missing/invalid/-1 => -1 (unlabeled)
    processed_labels = []
    unlabeled_count = 0
    invalid_count = 0

    for val in df["label"]:
        if pd.isna(val) or val is None or str(val).strip().lower() in ["", "nan", "none", "-1", "?"]:
            processed_labels.append(-1)
            unlabeled_count += 1
        else:
            try:
                num = int(float(val))
                if num in [0, 1]:
                    processed_labels.append(num)
                else:
                    # Treat unknown label value as unlabeled
                    processed_labels.append(-1)
                    invalid_count += 1
            except (ValueError, TypeError):
                processed_labels.append(-1)
                invalid_count += 1

    df["label"] = processed_labels
    clean_df = df[["node_id", "label"]].reset_index(drop=True)

    unknown_nodes_count = 0
    if known_nodes is not None:
        unknown_nodes_count = int((~clean_df["node_id"].isin(known_nodes)).sum())

    normal_count = int((clean_df["label"] == 0).sum())
    sybil_count = int((clean_df["label"] == 1).sum())
    explicit_unlabeled = int((clean_df["label"] == -1).sum())

    report = {
        "raw_label_rows": raw_count,
        "valid_node_ids": len(clean_df),
        "duplicate_nodes_dropped": dedup_labels,
        "normal_count": normal_count,
        "sybil_count": sybil_count,
        "unlabeled_count": explicit_unlabeled,
        "invalid_label_values": invalid_count,
        "nodes_not_in_graph": unknown_nodes_count,
    }

    return clean_df, report


def compute_graph_statistics(G: nx.Graph, labels_dict: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Compute comprehensive structural and label statistics for a NetworkX graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "max_degree": 0,
            "min_degree": 0,
            "num_components": 0,
            "isolated_nodes": 0,
            "largest_component_size": 0,
            "normal_count": 0,
            "sybil_count": 0,
            "unlabeled_count": 0,
            "sybil_ratio": 0.0,
        }

    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    max_degree = int(np.max(degrees)) if degrees else 0
    min_degree = int(np.min(degrees)) if degrees else 0
    isolated_nodes = sum(1 for d in degrees if d == 0)

    # Density
    possible_edges = (num_nodes * (num_nodes - 1)) / 2 if not G.is_directed() else (num_nodes * (num_nodes - 1))
    density = float(num_edges / possible_edges) if possible_edges > 0 else 0.0

    # Connected components
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    num_components = len(components)
    largest_component_size = max(len(c) for c in components) if components else 0

    # Label statistics
    normal_count = 0
    sybil_count = 0
    unlabeled_count = 0

    if isinstance(labels_dict, pd.DataFrame):
        labels_dict = dict(zip(labels_dict["node_id"].astype(str), labels_dict["label"].astype(int)))

    if labels_dict is not None and len(labels_dict) > 0:
        for node in G.nodes():
            lbl = labels_dict.get(str(node), -1)
            if lbl == 0:
                normal_count += 1
            elif lbl == 1:
                sybil_count += 1
            else:
                unlabeled_count += 1
    else:
        for node, data in G.nodes(data=True):
            lbl = data.get("label", -1)
            if lbl == 0:
                normal_count += 1
            elif lbl == 1:
                sybil_count += 1
            else:
                unlabeled_count += 1

    labeled_total = normal_count + sybil_count
    sybil_ratio = float(sybil_count / labeled_total) if labeled_total > 0 else 0.0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "min_degree": min_degree,
        "median_degree": float(np.median(degrees)) if degrees else 0.0,
        "num_components": num_components,
        "isolated_nodes": isolated_nodes,
        "largest_component_size": largest_component_size,
        "normal_count": normal_count,
        "sybil_count": sybil_count,
        "unlabeled_count": unlabeled_count,
        "labeled_count": labeled_total,
        "sybil_ratio_among_labeled": round(sybil_ratio, 4),
    }
