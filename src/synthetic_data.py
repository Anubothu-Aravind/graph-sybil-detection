"""Configurable synthetic social network generation for Sybil detection benchmarking."""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import networkx as nx

from src.utils import set_seed


def generate_synthetic_social_network(
    num_nodes: int = 1000,
    sybil_ratio: float = 0.20,
    labeled_ratio: float = 0.20,
    sybil_topology: str = "clustered",
    attack_edges_count: Optional[int] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate a realistic synthetic social network with honest communities and a Sybil region.

    Graph Topology:
      - Normal region: Multi-community small-world graph with high clustering and intra-cluster density.
      - Sybil region: Dense malicious cluster coordinated by an adversary.
      - Attack edges (Cut): Constrained number of bridging edges from Sybil accounts to honest accounts,
        reflecting classic Sybil defense literature (bounded quotient cut).

    Parameters:
        num_nodes: Total number of nodes in the graph (default: 1000).
        sybil_ratio: Fraction of nodes that are Sybil (malicious). Range (0.05, 0.50).
        labeled_ratio: Fraction of nodes with known training labels. Range (0.01, 0.50).
        sybil_topology: 'clustered' (dense Sybil cell), 'random' (Erdos-Renyi), or 'star' (central bot master).
        attack_edges_count: Explicit count of cross-region bridging edges. If None, set to ~5-15 edges.
        seed: Random seed for exact reproducibility.

    Returns:
        edges_df: DataFrame with ['source', 'target']
        labels_df: DataFrame with ['node_id', 'label'] where unlabeled nodes have label -1
        ground_truth_df: DataFrame with ['node_id', 'ground_truth'] (0 = Normal, 1 = Sybil)
        metadata: Dictionary with topology generation details
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    num_sybils = max(3, int(num_nodes * sybil_ratio))
    num_normal = num_nodes - num_sybils

    normal_nodes = [f"N_{i}" for i in range(num_normal)]
    sybil_nodes = [f"S_{j}" for j in range(num_sybils)]

    G = nx.Graph()
    G.add_nodes_from(normal_nodes, label=0, is_sybil=0)
    G.add_nodes_from(sybil_nodes, label=1, is_sybil=1)

    # 1. Construct Normal Network with 4-6 interconnected communities
    num_communities = max(2, min(8, num_normal // 150))
    community_size = num_normal // num_communities
    communities = []

    for c in range(num_communities):
        start_idx = c * community_size
        end_idx = (c + 1) * community_size if c < num_communities - 1 else num_normal
        c_nodes = normal_nodes[start_idx:end_idx]
        communities.append(c_nodes)

        # Average degree in each honest community target ~12-14 to achieve 5,000-10,000 edges for 1000 nodes
        target_k = min(14, max(4, len(c_nodes) // 8))
        if target_k % 2 != 0:
            target_k = max(2, target_k - 1)
        if len(c_nodes) > target_k:
            c_sub = nx.connected_watts_strogatz_graph(len(c_nodes), k=target_k, p=0.25, seed=int(rng.integers(1, 10000)))
            mapping = {i: c_nodes[i] for i in range(len(c_nodes))}
            c_sub = nx.relabel_nodes(c_sub, mapping)
            G.add_edges_from(c_sub.edges())

    # Add inter-community bridging edges among normal communities
    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            bridges = max(2, int(len(communities[i]) * 0.05))
            for _ in range(bridges):
                u = rng.choice(communities[i])
                v = rng.choice(communities[j])
                G.add_edge(u, v)

    # 2. Construct Sybil Region
    if sybil_topology == "star":
        # Bot master star topology
        master = sybil_nodes[0]
        for s in sybil_nodes[1:]:
            G.add_edge(master, s)
        extra_bot_edges = int(num_sybils * 1.5)
        for _ in range(extra_bot_edges):
            u = rng.choice(sybil_nodes[1:])
            v = rng.choice(sybil_nodes[1:])
            if u != v:
                G.add_edge(u, v)
    elif sybil_topology == "random":
        # Erdos-Renyi inside Sybils
        p_sybil = min(0.2, max(0.04, 12.0 / max(1, num_sybils)))
        s_sub = nx.erdos_renyi_graph(num_sybils, p=p_sybil, seed=int(rng.integers(1, 10000)))
        s_mapping = {i: sybil_nodes[i] for i in range(num_sybils)}
        s_sub = nx.relabel_nodes(s_sub, s_mapping)
        G.add_edges_from(s_sub.edges())
    else:
        # Default: Clustered (high density internal collusive network)
        k_s = min(16, max(4, num_sybils // 6))
        if k_s % 2 != 0:
            k_s = max(2, k_s - 1)
        if num_sybils > k_s:
            s_sub = nx.connected_watts_strogatz_graph(num_sybils, k=k_s, p=0.15, seed=int(rng.integers(1, 10000)))
            s_mapping = {i: sybil_nodes[i] for i in range(num_sybils)}
            s_sub = nx.relabel_nodes(s_sub, s_mapping)
            G.add_edges_from(s_sub.edges())
        else:
            for i in range(num_sybils):
                for j in range(i + 1, num_sybils):
                    G.add_edge(sybil_nodes[i], sybil_nodes[j])

    # 3. Add Bottleneck Attack Edges (Cut between Sybil and Normal)
    if attack_edges_count is None:
        # Tight quotient cut: 6 to 15 bridging connections
        attack_edges_count = max(4, int(num_sybils * 0.06))

    for _ in range(attack_edges_count):
        s_node = rng.choice(sybil_nodes)
        n_node = rng.choice(normal_nodes)
        G.add_edge(s_node, n_node)

    # Ensure full connectivity: connect any isolated nodes
    for comp in list(nx.connected_components(G)):
        if len(comp) < 3:
            for isolated in comp:
                target = rng.choice(normal_nodes if isolated in normal_nodes else sybil_nodes)
                if target != isolated:
                    G.add_edge(isolated, target)

    # 4. Formulate Edge List
    edges_list = list(G.edges())
    edges_df = pd.DataFrame(edges_list, columns=["source", "target"])

    # 5. Formulate Ground Truth and Semi-Supervised Labels
    all_nodes = list(G.nodes())
    ground_truth = {node: (1 if node.startswith("S_") else 0) for node in all_nodes}

    gt_df = pd.DataFrame([
        {"node_id": node, "ground_truth": ground_truth[node]}
        for node in all_nodes
    ])

    # Select labeled subset
    total_labeled = max(2, int(len(all_nodes) * labeled_ratio))
    labeled_normal_count = max(1, int(total_labeled * (1.0 - sybil_ratio)))
    labeled_sybil_count = max(1, total_labeled - labeled_normal_count)

    labeled_normal = rng.choice(normal_nodes, size=min(labeled_normal_count, len(normal_nodes)), replace=False)
    labeled_sybils = rng.choice(sybil_nodes, size=min(labeled_sybil_count, len(sybil_nodes)), replace=False)

    known_labels = {}
    for node in all_nodes:
        if node in labeled_normal:
            known_labels[node] = 0
        elif node in labeled_sybils:
            known_labels[node] = 1
        else:
            known_labels[node] = -1

    labels_df = pd.DataFrame([
        {"node_id": node, "label": known_labels[node]}
        for node in all_nodes
    ])

    metadata = {
        "num_total_nodes": len(all_nodes),
        "num_normal_nodes": num_normal,
        "num_sybil_nodes": num_sybils,
        "sybil_ratio_actual": round(num_sybils / len(all_nodes), 4),
        "num_edges": len(edges_df),
        "num_attack_edges": attack_edges_count,
        "num_labeled_nodes": len(labeled_normal) + len(labeled_sybils),
        "labeled_ratio_actual": round((len(labeled_normal) + len(labeled_sybils)) / len(all_nodes), 4),
        "labeled_normal": len(labeled_normal),
        "labeled_sybil": len(labeled_sybils),
        "seed": seed,
        "sybil_topology": sybil_topology,
        "is_synthetic": True,
        "notice": "Simulated synthetic social network for benchmark and evaluation."
    }

    return edges_df, labels_df, gt_df, metadata


def generate_demo_dataset(target_dir: str = "data") -> Tuple[str, str]:
    """Generate default demo dataset files in data/ directory for instant out-of-the-box usage."""
    os.makedirs(target_dir, exist_ok=True)
    edges_path = os.path.join(target_dir, "demo_edges.csv")
    labels_path = os.path.join(target_dir, "demo_labels.csv")

    edges_df, labels_df, _, _ = generate_synthetic_social_network(
        num_nodes=1000,
        sybil_ratio=0.20,
        labeled_ratio=0.20,
        sybil_topology="clustered",
        attack_edges_count=12,
        seed=42
    )

    edges_df.to_csv(edges_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    return edges_path, labels_path
