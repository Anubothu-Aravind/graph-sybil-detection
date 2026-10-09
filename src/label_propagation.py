"""Semi-supervised Graph Label Propagation algorithms for Sybil detection.

Includes:
- GraphShieldLabelPropagation: Custom matrix-based iterative label propagation
  supporting both Random-Walk and Symmetric Laplacian normalization, hard and soft clamping,
  configurable propagation parameter alpha, and optional Apple MPS / PyTorch GPU acceleration.
- BaselineLabelSpreading: scikit-learn LabelSpreading baseline wrapper.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

from src.graph_builder import graph_to_sparse_adjacency
from src.utils import get_torch_device


class GraphShieldLabelPropagation:
    """Custom Semi-Supervised Graph Label Propagation for Sybil detection.

    Mathematical Formulation:
      Given graph G = (V, E) with adjacency matrix A and degree matrix D:
      1. Normalized propagation operator:
         - 'random_walk': P = D^{-1} A (row-stochastic transition matrix)
         - 'normalized_laplacian': P = D^{-1/2} A D^{-1/2} (symmetric normalized)
      2. Initial label matrix Y^{(0)} in R^{n x 2}:
         - Labeled Normal: [1.0, 0.0]
         - Labeled Sybil:  [0.0, 1.0]
         - Unlabeled:      [0.5, 0.5] (or empirical prior)
      3. Iterative propagation:
         Y^{(t+1)} = alpha * P * Y^{(t)} + (1 - alpha) * Y^{(0)}
      4. Clamping:
         If clamping=True: Y_L^{(t+1)} = Y_L^{(0)} (hard boundary condition)
      5. Convergence:
         Stop when ||Y^{(t+1)} - Y^{(t)}||_F < tol or t == max_iter.
      6. Decision rule:
         Confidence p(Sybil) = Y_{i, 1} / (Y_{i, 0} + Y_{i, 1})
         Predicted Sybil if p(Sybil) >= threshold.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-4,
        clamping: bool = True,
        normalization: str = "random_walk",
        threshold: float = 0.50,
        use_torch: bool = False,
        device: str = "cpu"
    ):
        """Initialize GraphShield label propagation model.

        Args:
            alpha: Propagation parameter in (0, 1). Weight on propagated neighbor scores vs initial state.
            max_iter: Maximum propagation iterations.
            tol: Frobenius norm convergence tolerance.
            clamping: If True, keep known training labels strictly fixed (harmonic function / hard clamping).
            normalization: 'random_walk' (D^-1 A) or 'normalized_laplacian' (D^-1/2 A D^-1/2).
            threshold: Decision threshold for classifying a node as Sybil (>= threshold).
            use_torch: Whether to use PyTorch backend (supports Apple MPS acceleration).
            device: 'cpu' or 'mps'.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.clamping = bool(clamping)
        self.normalization = normalization
        self.threshold = float(threshold)
        self.use_torch = bool(use_torch)
        self.device = device

        # Run statistics and state
        self.iterations_run_ = 0
        self.converged_ = False
        self.final_loss_ = 0.0
        self.nodes_ = []
        self.node_to_idx_ = {}
        self.probabilities_ = None
        self.predictions_ = None
        self.confidence_scores_ = None

    def _compute_propagation_matrix(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """Compute normalized propagation matrix P from adjacency matrix A."""
        n = A.shape[0]
        deg = np.array(A.sum(axis=1)).flatten()

        # Handle isolated nodes (degree = 0) by adding self-loop so row doesn't vanish
        zero_deg = np.where(deg == 0)[0]
        if len(zero_deg) > 0:
            A = A.tolil()
            for idx in zero_deg:
                A[idx, idx] = 1.0
            A = A.tocsr()
            deg = np.array(A.sum(axis=1)).flatten()

        if self.normalization == "normalized_laplacian":
            # D^{-1/2} A D^{-1/2}
            deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
            nonzero = deg > 0
            deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
            D_inv_sqrt = sp.diags(deg_inv_sqrt)
            P = D_inv_sqrt @ A @ D_inv_sqrt
        else:
            # Default: D^{-1} A (Random Walk)
            deg_inv = np.zeros_like(deg, dtype=np.float32)
            nonzero = deg > 0
            deg_inv[nonzero] = 1.0 / deg[nonzero]
            D_inv = sp.diags(deg_inv)
            P = D_inv @ A

        return P.tocsr()

    def fit_predict(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        known_labels: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Execute label propagation on graph G.

        Args:
            G: NetworkX graph.
            known_labels: Dict mapping node_id -> label (0=Normal, 1=Sybil, -1=Unlabeled).
                          If None, labels will be read from G.nodes[n]['label'].

        Returns:
            Dictionary containing node-level predictions, probabilities, and execution metadata.
        """
        A, nodes, node_to_idx = graph_to_sparse_adjacency(G)
        self.nodes_ = nodes
        self.node_to_idx_ = node_to_idx
        n = len(nodes)

        if n == 0:
            return {"nodes": [], "predictions": {}, "confidences": {}, "probabilities": {}}

        # Detect classes
        valid_classes = sorted(list(set(lbl for lbl in known_labels.values() if lbl != -1)))
        if not valid_classes:
            return {
                "predictions": {n: 0 for n in nodes},
                "confidence_scores": {n: 0.5 for n in nodes},
                "probabilities": {n: [0.5, 0.5] for n in nodes},
                "iterations": 0,
                "converged": True,
                "final_delta": 0.0,
                "clamping": self.clamping,
                "alpha": self.alpha,
                "threshold": self.threshold,
                "normalization": self.normalization,
                "backend": "scipy_numpy"
            }

        num_classes = max(2, max(valid_classes) + 1)
        is_binary = (num_classes == 2)

        # Build initial label matrix Y0 in R^{n x K}
        Y0 = np.full((n, num_classes), 1.0 / num_classes, dtype=np.float32)
        labeled_mask = np.zeros(n, dtype=bool)

        for i, node in enumerate(nodes):
            lbl = known_labels.get(node, -1)
            if lbl != -1 and 0 <= lbl < num_classes:
                Y0[i] = 0.0
                Y0[i, lbl] = 1.0
                labeled_mask[i] = True

        P = self._compute_propagation_matrix(A)

        # Execute propagation
        if self.use_torch and self.device == "mps":
            Y_final, iters, converged, diff = self._propagate_torch(P, Y0, labeled_mask, "mps")
        else:
            Y_final, iters, converged, diff = self._propagate_numpy(P, Y0, labeled_mask)

        self.iterations_run_ = iters
        self.converged_ = converged
        self.final_loss_ = diff

        # Normalize row-wise to get probabilities
        row_sums = Y_final.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = Y_final / row_sums

        if is_binary:
            sybil_conf = probs[:, 1]
            preds = (sybil_conf >= self.threshold).astype(int)
            if self.clamping:
                for i in range(n):
                    if labeled_mask[i]:
                        orig_lbl = 1 if Y0[i, 1] > 0.5 else 0
                        preds[i] = orig_lbl
                        sybil_conf[i] = 1.0 if orig_lbl == 1 else 0.0
            confidences = {nodes[i]: float(np.round(sybil_conf[i], 4)) for i in range(n)}
        else:
            preds = np.argmax(probs, axis=1)
            max_conf = np.max(probs, axis=1)
            if self.clamping:
                for i in range(n):
                    if labeled_mask[i]:
                        orig_lbl = int(np.argmax(Y0[i]))
                        preds[i] = orig_lbl
                        max_conf[i] = 1.0
            confidences = {nodes[i]: float(np.round(max_conf[i], 4)) for i in range(n)}

        # Construct outputs
        results = {
            "predictions": {nodes[i]: int(preds[i]) for i in range(n)},
            "confidence_scores": confidences,
            "probabilities": {nodes[i]: [float(np.round(p, 4)) for p in probs[i]] for i in range(n)},
            "iterations": iters,
            "converged": converged,
            "final_delta": float(diff),
            "clamping": self.clamping,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "normalization": self.normalization,
            "num_classes": num_classes,
            "is_binary": is_binary,
            "backend": "torch_mps" if (self.use_torch and self.device == "mps") else "scipy_numpy",
        }

        self.probabilities_ = results["probabilities"]
        self.predictions_ = results["predictions"]
        self.confidence_scores_ = results["confidence_scores"]

        return results

    def _propagate_numpy(
        self,
        P: sp.csr_matrix,
        Y0: np.ndarray,
        labeled_mask: np.ndarray
    ) -> Tuple[np.ndarray, int, bool, float]:
        """NumPy / SciPy sparse iterative propagation."""
        Y = Y0.copy()
        converged = False
        diff = float("inf")
        t = 0

        for t in range(1, self.max_iter + 1):
            # Y_{new} = alpha * P * Y + (1 - alpha) * Y0
            Y_new = self.alpha * (P @ Y) + (1.0 - self.alpha) * Y0

            if self.clamping:
                Y_new[labeled_mask] = Y0[labeled_mask]

            diff = float(np.linalg.norm(Y_new - Y, ord="fro"))
            Y = Y_new

            if diff < self.tol:
                converged = True
                break

        return Y, t, converged, diff

    def _propagate_torch(
        self,
        P: sp.csr_matrix,
        Y0: np.ndarray,
        labeled_mask: np.ndarray,
        device_str: str = "mps"
    ) -> Tuple[np.ndarray, int, bool, float]:
        """PyTorch Metal Performance Shaders (MPS) or GPU accelerated propagation."""
        import torch

        dev = torch.device(device_str)
        coo = P.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        # Explicitly opt out of invariant checking to silence warning and maximize throughput
        P_t = torch.sparse_coo_tensor(
            indices, values, size=coo.shape, device=dev, is_coalesced=True, check_invariants=False
        )

        Y0_t = torch.tensor(Y0, dtype=torch.float32, device=dev)
        Y_t = Y0_t.clone()
        mask_t = torch.tensor(labeled_mask, dtype=torch.bool, device=dev)

        alpha_t = float(self.alpha)
        one_minus_alpha = 1.0 - alpha_t
        converged = False
        diff = float("inf")
        t = 0

        for t in range(1, self.max_iter + 1):
            Y_new_t = alpha_t * torch.sparse.mm(P_t, Y_t) + one_minus_alpha * Y0_t

            if self.clamping:
                Y_new_t[mask_t] = Y0_t[mask_t]

            diff_t = torch.norm(Y_new_t - Y_t)
            diff = float(diff_t.item())
            Y_t = Y_new_t

            if diff < self.tol:
                converged = True
                break

        Y_np = Y_t.cpu().numpy()
        return Y_np, t, converged, diff


class BaselineLabelSpreading:
    """Established baseline using scikit-learn's LabelSpreading on graph adjacency matrix."""

    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.model = LabelSpreading(
            kernel=lambda x, y=None: x,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol
        )
        self.nodes_ = []

    def fit_predict(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        known_labels: Dict[str, int]
    ) -> Dict[str, Any]:
        """Fit scikit-learn LabelSpreading baseline and predict on graph nodes."""
        import warnings

        A, nodes, _ = graph_to_sparse_adjacency(G)
        self.nodes_ = nodes
        n = len(nodes)

        # Scikit-learn LabelSpreading expects CSR sparse or dense affinity matrix
        # Add self-loops to diagonal for graph affinity
        affinity = A.tocsr().astype(np.float32)
        affinity.setdiag(1.0)

        y = np.array([known_labels.get(node, -1) for node in nodes], dtype=int)

        # Check if we have at least 2 distinct labeled classes
        labeled_mask = (y != -1)
        classes_present = np.unique(y[labeled_mask]) if np.any(labeled_mask) else np.array([])

        if len(classes_present) < 2:
            default_cls = int(classes_present[0]) if len(classes_present) == 1 else 0
            return {
                "predictions": {node: default_cls for node in nodes},
                "confidence_scores": {node: 0.5 for node in nodes},
                "probabilities": {node: [0.5, 0.5] for node in nodes},
                "iterations": 0,
                "algorithm": "scikit-learn LabelSpreading (Degenerate Class Fallback)"
            }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            self.model.fit(affinity, y)
            probs = self.model.predict_proba(affinity)
            preds = self.model.predict(affinity)

        # Clean any potential NaN values from disconnected components
        probs = np.nan_to_num(probs, nan=1.0 / max(1, probs.shape[1]))
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums

        classes = list(self.model.classes_)
        is_binary = len(classes) == 2 and set(classes).issubset({0, 1})

        if is_binary:
            sybil_col = classes.index(1) if 1 in classes else 0
            confidences = {nodes[i]: float(np.round(probs[i, sybil_col], 4)) for i in range(n)}
        else:
            max_conf = np.max(probs, axis=1)
            confidences = {nodes[i]: float(np.round(max_conf[i], 4)) for i in range(n)}

        predictions = {nodes[i]: int(preds[i]) for i in range(n)}
        probabilities = {
            nodes[i]: [float(np.round(p, 4)) for p in probs[i]]
            for i in range(n)
        }

        return {
            "predictions": predictions,
            "confidence_scores": confidences,
            "probabilities": probabilities,
            "iterations": getattr(self.model, "n_iter_", self.max_iter),
            "algorithm": "scikit-learn LabelSpreading"
        }
