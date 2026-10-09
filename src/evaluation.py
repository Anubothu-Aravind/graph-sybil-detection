"""Evaluation pipeline with withheld test evaluation, leakage prevention, and baseline comparison."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.utils import set_seed
from src.label_propagation import GraphShieldLabelPropagation, BaselineLabelSpreading


def split_train_test_labels(
    ground_truth: Dict[str, int],
    train_ratio: float = 0.30,
    seed: int = 42
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Partition ground-truth labeled nodes into training and withheld test sets.

    Guarantees zero test-leakage:
      - 'train_labels' contains labels for only train-partitioned nodes, with all other nodes set to -1.
      - 'test_labels' contains true labels for only held-out test nodes.

    Args:
        ground_truth: Mapping of node_id -> true integer label (>= 0).
        train_ratio: Fraction of labeled nodes provided as training seeds (e.g. 0.1 to 0.5).
        seed: Random seed for reproducibility.

    Returns:
        train_labels: Dict of all nodes, where training nodes have 0/1/etc. and test nodes have -1.
        test_labels: Dict of only held-out test nodes with their ground-truth labels.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Filter strictly valid ground-truth nodes (>= 0)
    valid_nodes = [n for n, l in ground_truth.items() if l is not None and l >= 0]
    classes = sorted(list(set(ground_truth[n] for n in valid_nodes)))

    train_set = set()
    for c in classes:
        c_nodes = [n for n in valid_nodes if ground_truth[n] == c]
        if c_nodes:
            n_train_c = max(1, int(len(c_nodes) * train_ratio))
            chosen = rng.choice(c_nodes, size=min(n_train_c, len(c_nodes)), replace=False)
            train_set.update(chosen)

    train_labels = {}
    test_labels = {}

    for node in ground_truth.keys():
        if node in train_set:
            train_labels[node] = ground_truth[node]
        else:
            train_labels[node] = -1
            if node in valid_nodes:
                test_labels[node] = ground_truth[node]

    return train_labels, test_labels


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Positive class = 1 (Sybil). Negative class = 0 (Normal).
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "specificity": 0.0,
            "roc_auc": 0.0,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0],
            "test_sample_count": 0,
        }

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    classes = np.unique(np.concatenate([y_t, y_p]))
    is_binary = len(classes) <= 2 and set(classes).issubset({0, 1})

    acc = float(accuracy_score(y_t, y_p))

    if is_binary:
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        auc = 0.5
        fpr_list, tpr_list = [0.0, 1.0], [0.0, 1.0]

        if y_scores is not None and len(np.unique(y_t)) > 1:
            try:
                auc = float(roc_auc_score(y_t, y_scores))
                fpr, tpr, _ = roc_curve(y_t, y_scores)
                fpr_list = [float(x) for x in fpr]
                tpr_list = [float(x) for x in tpr]
            except Exception:
                auc = 0.5

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "cm_matrix": [[tn, fp], [fn, tp]],
            "fpr": fpr_list,
            "tpr": tpr_list,
            "test_sample_count": len(y_t),
            "is_binary": True,
        }
    else:
        # Multi-class evaluation (e.g. Cora, PubMed)
        prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
        rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
        f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        cm = confusion_matrix(y_t, y_p)

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "specificity": 0.0,
            "roc_auc": 0.0,
            "confusion_matrix": {"raw_cm": cm.tolist()},
            "cm_matrix": cm.tolist(),
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0],
            "test_sample_count": len(y_t),
            "is_binary": False,
        }


def evaluate_detection(
    model: Any,
    G: Any,
    train_labels: Dict[str, int],
    test_labels: Dict[str, int]
) -> Dict[str, Any]:
    """Execute model training on train_labels and evaluate on held-out test_labels."""
    results = model.fit_predict(G, known_labels=train_labels)

    preds_dict = results["predictions"]
    conf_dict = results["confidence_scores"]

    test_nodes = sorted(list(test_labels.keys()))
    y_true = [test_labels[n] for n in test_nodes]
    y_pred = [preds_dict[n] for n in test_nodes]
    y_scores = [conf_dict[n] for n in test_nodes]

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    metrics["test_nodes"] = test_nodes
    metrics["predictions"] = {n: preds_dict[n] for n in test_nodes}
    metrics["confidences"] = {n: conf_dict[n] for n in test_nodes}
    metrics["iterations"] = results.get("iterations", 0)
    metrics["converged"] = results.get("converged", True)

    return metrics


def run_benchmark_comparison(
    G: Any,
    ground_truth: Dict[str, int],
    train_ratio: float = 0.30,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-4,
    clamping: bool = True,
    threshold: float = 0.50,
    seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Run side-by-side benchmark comparison between GraphShield and Scikit-learn Baseline."""
    train_labels, test_labels = split_train_test_labels(
        ground_truth, train_ratio=train_ratio, seed=seed
    )

    # 1. Custom GraphShield Model
    custom_model = GraphShieldLabelPropagation(
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        clamping=clamping,
        threshold=threshold,
    )
    custom_metrics = evaluate_detection(custom_model, G, train_labels, test_labels)

    # 2. Baseline Model
    baseline_model = BaselineLabelSpreading(
        alpha=alpha,
        max_iter=min(50, max_iter),
        tol=max(1e-3, tol)
    )
    baseline_metrics = evaluate_detection(baseline_model, G, train_labels, test_labels)

    is_bin = custom_metrics.get("is_binary", True)

    if is_bin:
        df_compare = pd.DataFrame([
            {
                "Metric": "Accuracy",
                "GraphShield (Custom LP)": custom_metrics["accuracy"],
                "Baseline (LabelSpreading)": baseline_metrics["accuracy"],
            },
            {
                "Metric": "Precision (Sybil)",
                "GraphShield (Custom LP)": custom_metrics["precision"],
                "Baseline (LabelSpreading)": baseline_metrics["precision"],
            },
            {
                "Metric": "Recall (Sybil)",
                "GraphShield (Custom LP)": custom_metrics["recall"],
                "Baseline (LabelSpreading)": baseline_metrics["recall"],
            },
            {
                "Metric": "F1-Score",
                "GraphShield (Custom LP)": custom_metrics["f1_score"],
                "Baseline (LabelSpreading)": baseline_metrics["f1_score"],
            },
            {
                "Metric": "ROC-AUC",
                "GraphShield (Custom LP)": custom_metrics.get("roc_auc", 0.5),
                "Baseline (LabelSpreading)": baseline_metrics.get("roc_auc", 0.5),
            },
            {
                "Metric": "Iterations Run",
                "GraphShield (Custom LP)": custom_metrics["iterations"],
                "Baseline (LabelSpreading)": baseline_metrics["iterations"],
            },
        ])
    else:
        df_compare = pd.DataFrame([
            {
                "Metric": "Accuracy",
                "GraphShield (Custom LP)": custom_metrics["accuracy"],
                "Baseline (LabelSpreading)": baseline_metrics["accuracy"],
            },
            {
                "Metric": "Macro Precision",
                "GraphShield (Custom LP)": custom_metrics["precision"],
                "Baseline (LabelSpreading)": baseline_metrics["precision"],
            },
            {
                "Metric": "Macro Recall",
                "GraphShield (Custom LP)": custom_metrics["recall"],
                "Baseline (LabelSpreading)": baseline_metrics["recall"],
            },
            {
                "Metric": "Macro F1-Score",
                "GraphShield (Custom LP)": custom_metrics["f1_score"],
                "Baseline (LabelSpreading)": baseline_metrics["f1_score"],
            },
            {
                "Metric": "Iterations Run",
                "GraphShield (Custom LP)": custom_metrics["iterations"],
                "Baseline (LabelSpreading)": baseline_metrics["iterations"],
            },
        ])

    return custom_metrics, baseline_metrics, df_compare
