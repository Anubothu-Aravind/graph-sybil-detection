"""Unit tests for src/synthetic_data.py."""

import os
import pytest
import pandas as pd
import networkx as nx

from src.synthetic_data import generate_synthetic_social_network, generate_demo_dataset


def test_generate_synthetic_social_network_structure():
    edges_df, labels_df, gt_df, meta = generate_synthetic_social_network(
        num_nodes=80,
        sybil_ratio=0.25,
        labeled_ratio=0.20,
        sybil_topology="clustered",
        seed=123
    )

    assert len(gt_df) == 80
    assert len(labels_df) == 80
    assert meta["num_sybil_nodes"] == 20
    assert meta["num_normal_nodes"] == 60

    # Verify edge connectivity
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"])

    assert G.number_of_nodes() == 80
    assert nx.is_connected(G)


def test_generate_synthetic_reproducibility():
    e1, l1, gt1, _ = generate_synthetic_social_network(num_nodes=50, seed=42)
    e2, l2, gt2, _ = generate_synthetic_social_network(num_nodes=50, seed=42)

    pd.testing.assert_frame_equal(e1, e2)
    pd.testing.assert_frame_equal(l1, l2)
    pd.testing.assert_frame_equal(gt1, gt2)


def test_generate_demo_dataset(tmp_path):
    target = str(tmp_path)
    e_path, l_path = generate_demo_dataset(target)
    assert os.path.exists(e_path)
    assert os.path.exists(l_path)
    df_e = pd.read_csv(e_path)
    df_l = pd.read_csv(l_path)
    assert len(df_e) > 0
    assert len(df_l) > 0
