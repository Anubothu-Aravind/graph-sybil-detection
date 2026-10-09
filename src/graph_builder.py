"""Graph construction, feature extraction, and structural processing using NetworkX."""

from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

from src.data_loader import load_and_validate_edges, load_and_validate_labels


def build_networkx_graph(
    edges_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    directed: bool = False,
    remove_self_loops: bool = True
) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[str, Any]]:
    """Construct a NetworkX graph from validated edges and node labels.

    Args:
        edges_df: DataFrame with ['source', 'target']
        labels_df: Optional DataFrame with ['node_id', 'label'] (0=Normal, 1=Sybil, -1=Unlabeled)
        directed: Whether to construct a directed or undirected graph
        remove_self_loops: Whether to strip self-loops (u == v)

    Returns:
        G: NetworkX Graph or DiGraph
        info: Summary of graph construction (node count, edge count, isolated nodes, components)
    """
    G = nx.DiGraph() if directed else nx.Graph()

    # Add edges
    for _, row in edges_df.iterrows():
        u = str(row["source"]).strip()
        v = str(row["target"]).strip()
        if remove_self_loops and u == v:
            continue
        G.add_edge(u, v)

    # Attach labels from labels_df
    labels_dict: Dict[str, int] = {}
    if labels_df is not None and not labels_df.empty:
        for _, row in labels_df.iterrows():
            nid = str(row["node_id"]).strip()
            lbl = int(row["label"])
            labels_dict[nid] = lbl
            # If node had no edges, it is added as an isolated node
            if nid not in G:
                G.add_node(nid)

    # Assign node attributes
    for node in G.nodes():
        lbl = labels_dict.get(node, -1)
        G.nodes[node]["node_id"] = node
        G.nodes[node]["label"] = lbl

    # Compute node features
    degrees = dict(G.degree())
    clustering = {}
    if not directed:
        try:
            clustering = nx.clustering(G)
        except Exception:
            clustering = {n: 0.0 for n in G.nodes()}

    # Compute suspicious neighbors count (number of known Sybil neighbors, label=1)
    suspicious_neighbors = {}
    for node in G.nodes():
        neighbors = G.neighbors(node)
        s_count = sum(1 for nbr in neighbors if G.nodes[nbr].get("label") == 1)
        suspicious_neighbors[node] = s_count

    for node in G.nodes():
        G.nodes[node]["degree"] = degrees.get(node, 0)
        G.nodes[node]["clustering"] = round(clustering.get(node, 0.0), 4)
        G.nodes[node]["known_suspicious_neighbors"] = suspicious_neighbors.get(node, 0)

    # Calculate connected components
    if directed:
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    isolated_nodes = [node for node, deg in degrees.items() if deg == 0]

    info = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": directed,
        "num_components": len(components),
        "largest_component_nodes": max(len(c) for c in components) if components else 0,
        "isolated_nodes_count": len(isolated_nodes),
        "isolated_nodes_list": isolated_nodes,
        "labeled_nodes_count": sum(1 for n in G.nodes() if G.nodes[n].get("label") in [0, 1]),
    }

    return G, info


def graph_to_sparse_adjacency(
    G: Union[nx.Graph, nx.DiGraph],
    node_order: Optional[List[str]] = None
) -> Tuple[sp.csr_matrix, List[str], Dict[str, int]]:
    """Convert NetworkX graph into a CSR sparse adjacency matrix and node index mappings.

    Returns:
        A: scipy.sparse.csr_matrix
        nodes: ordered list of node IDs matching matrix rows/cols
        node_to_idx: mapping from node ID to row/column index
    """
    if node_order is None:
        nodes = sorted(list(G.nodes()))
    else:
        nodes = list(node_order)

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    rows, cols, data = [], [], []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            if not G.is_directed() and i != j:
                rows.append(j)
                cols.append(i)
                data.append(1.0)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return A, nodes, node_to_idx
