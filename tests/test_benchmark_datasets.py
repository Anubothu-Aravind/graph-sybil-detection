"""Unit tests for src/benchmark_datasets.py."""

import pytest
import os
from src.benchmark_datasets import load_cora_dataset, load_pubmed_dataset


def test_load_cora_dataset():
    edges_df, labels_df, gt_df, meta = load_cora_dataset("data/planetoid")

    assert len(edges_df) > 5000
    assert len(gt_df) == 2708
    assert len(labels_df) == 2708
    assert meta["num_classes"] == 7
    assert meta["is_citation_benchmark"] is True
    assert "source" in edges_df.columns
    assert "target" in edges_df.columns
    assert "node_id" in labels_df.columns
    assert "label" in labels_df.columns


def test_load_pubmed_dataset():
    edges_df, labels_df, gt_df, meta = load_pubmed_dataset("data/planetoid")

    assert len(edges_df) > 40000
    assert len(gt_df) == 19717
    assert meta["num_classes"] == 3
    assert meta["is_citation_benchmark"] is True
