# GraphShield: Benchmark Artifacts & Output Directory

This directory contains key outputs, high-resolution publication charts, and standalone reference code generated for the **GraphShield Sybil Detection** project report.

---

## 1. High-Resolution Visualizations (300 DPI PNG)

| File | Description |
| :--- | :--- |
| `01_confusion_matrix_synthetic.png` | Confusion matrix of held-out test accounts (Honest vs Sybil) showing zero false positives/negatives under bounded cut. |
| `02_roc_curve_synthetic.png` | Receiver Operating Characteristic (ROC) curve showing True Positive Rate vs False Positive Rate (AUC = 1.0000). |
| `03_degree_distribution_synthetic.png` | Histogram of node connectivity degrees across the 1,000-node synthetic benchmark network (mean degree = 14.6). |
| `04_benchmark_comparison_metrics.png` | Grouped bar chart comparing GraphShield vs Baseline across Accuracy and F1-Score on Synthetic, Cora, and PubMed networks. |
| `05_convergence_and_runtime.png` | Comparative bar plots of inference runtime (sub-second) and power iterations to convergence across all datasets. |
| `06_sybil_graph_topology.png` | Force-directed 2D network diagram visualizing Honest accounts (blue) vs Sybil adversarial cell (red) with bounded attack bridge. |

---

## 2. Experimental Data & Results

| File | Description | Format |
| :--- | :--- | :--- |
| `evaluation_summary.txt` | Formatted comparison table with exact Accuracy, Precision, Recall, F1, iterations, and runtimes across all benchmarks. | Plain Text Table |
| `benchmark_results.json` | Comprehensive machine-readable benchmark metrics and dataset dimensions. | JSON |
| `synthetic_predictions_sample.csv` | Sample predictions inventory (Node ID, Ground Truth, Predicted Class, Continuous Confidence, Node Degree). | CSV |

---

## 3. Important Implementation Code

| File | Description |
| :--- | :--- |
| `graphshield_core_engine.py` | Standalone, self-contained reference Python implementation of the sparse label propagation algorithm with Random Walk and Symmetric Laplacian transition operators. |
