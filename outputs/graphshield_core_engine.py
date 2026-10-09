"""GraphShield: Core Label Propagation & Sybil Detection Engine.

This standalone reference module encapsulates the core mathematical implementation
of GraphShield's semi-supervised sparse label propagation algorithm:
    Y^{(t+1)} = alpha * (P @ Y^{(t)}) + (1 - alpha) * Y^{(0)}
with hard prior clamping:
    Y^{(t+1)}_L = Y^{(0)}_L
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import scipy.sparse as sp
import networkx as nx


def normalize_transition_matrix(
    A: sp.csr_matrix,
    mode: str = "random_walk"
) -> sp.csr_matrix:
    """Normalize adjacency matrix into a stochastic transition matrix.
    
    Modes:
      - 'random_walk': P = D^{-1} A (row-stochastic)
      - 'symmetric':   P = D^{-1/2} A D^{-1/2} (Laplacian-normalized)
    """
    n = A.shape[0]
    # Add self-loops to isolated nodes to avoid zero division
    degrees = np.array(A.sum(axis=1)).flatten()
    isolated_idx = np.where(degrees == 0)[0]
    if len(isolated_idx) > 0:
        A = A.copy().tolil()
        for idx in isolated_idx:
            A[idx, idx] = 1.0
        A = A.tocsr()
        degrees = np.array(A.sum(axis=1)).flatten()

    if mode == "random_walk":
        inv_degrees = np.power(degrees, -1.0, where=(degrees > 0))
        inv_degrees[degrees == 0] = 0.0
        D_inv = sp.diags(inv_degrees, format="csr")
        return D_inv @ A
    elif mode == "symmetric":
        inv_sqrt = np.power(degrees, -0.5, where=(degrees > 0))
        inv_sqrt[degrees == 0] = 0.0
        D_inv_sqrt = sp.diags(inv_sqrt, format="csr")
        return D_inv_sqrt @ A @ D_inv_sqrt
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


class GraphShieldLabelPropagation:
    """Semi-Supervised Graph Label Propagation with Sparse Matrix Operations."""

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-3,
        clamping: bool = True,
        normalization: str = "random_walk",
        threshold: float = 0.50
    ):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.clamping = bool(clamping)
        self.normalization = normalization
        self.threshold = float(threshold)

    def fit_predict(
        self,
        G: nx.Graph,
        known_labels: Dict[str, int]
    ) -> Dict[str, Any]:
        """Fit model and return predicted classes and posterior continuous confidence scores."""
        nodes = sorted(list(G.nodes()))
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        # Build sparse adjacency matrix
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float32)
        P = normalize_transition_matrix(A, mode=self.normalization)

        # Detect number of classes K
        labeled_nodes = [n for n in nodes if known_labels.get(n, -1) != -1]
        classes_present = sorted(list({known_labels[n] for n in labeled_nodes})) if labeled_nodes else [0, 1]
        K = max(2, len(classes_present), (max(classes_present) + 1) if classes_present else 2)

        # Initialize prior matrix Y0
        Y0 = np.full((n, K), 1.0 / K, dtype=np.float32)
        labeled_mask = np.zeros(n, dtype=bool)

        for node_id, class_label in known_labels.items():
            if class_label != -1 and node_id in node_to_idx:
                idx = node_to_idx[node_id]
                labeled_mask[idx] = True
                Y0[idx, :] = 0.0
                if 0 <= class_label < K:
                    Y0[idx, class_label] = 1.0

        # Power iteration: Y^{(t+1)} = alpha * P @ Y + (1 - alpha) * Y0
        Y = Y0.copy()
        converged = False
        t = 0

        for t in range(1, self.max_iter + 1):
            Y_new = self.alpha * (P @ Y) + (1.0 - self.alpha) * Y0
            if self.clamping:
                Y_new[labeled_mask] = Y0[labeled_mask]

            diff = float(np.linalg.norm(Y_new - Y, ord="fro"))
            Y = Y_new
            if diff < self.tol:
                converged = True
                break

        # Normalize posterior probabilities
        row_sums = Y.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = Y / row_sums

        # Binary Sybil vs Honest decision thresholding
        is_binary = (K == 2)
        predictions = {}
        confidence_scores = {}

        if is_binary:
            for i, n in enumerate(nodes):
                conf = float(probs[i, 1])
                confidence_scores[n] = round(conf, 4)
                predictions[n] = 1 if conf >= self.threshold else 0
        else:
            for i, n in enumerate(nodes):
                pred_cls = int(np.argmax(probs[i]))
                predictions[n] = pred_cls
                confidence_scores[n] = round(float(np.max(probs[i])), 4)

        return {
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "iterations": t,
            "converged": converged,
            "total_nodes": n
        }
