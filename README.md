# GraphShield: Graph-Based Sybil Node Detection System

[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon%20%26%20Intel)-brightgreen.svg)](#macos-setup-requirements)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](#installation-steps)
[![Hardware: Apple Metal (MPS)](https://img.shields.io/badge/Hardware-Apple%20Metal%20MPS%20%2F%20CPU-orange.svg)](#apple-silicon-metal-performance-shaders-mps)
[![Tests: Passing](https://img.shields.io/badge/Tests-23%20Passed-success.svg)](#running-automated-tests)

**GraphShield** is an academic-grade Graph Machine Learning application that detects Sybil (fake and malicious) nodes in social and communication networks using semi-supervised graph label propagation.

Designed specifically for **macOS** (both Apple Silicon and Intel architectures), GraphShield operates locally without requiring external cloud infrastructure or proprietary CUDA hardware, leveraging **Apple Metal Performance Shaders (MPS)** unified memory acceleration with automatic SciPy/NumPy sparse CPU fallback.

---

## 5-Stage Pipeline Workflow

GraphShield organizes the complete detection and evaluation lifecycle into five modular stages accessible via the unified Streamlit interface:

```text
[Stage 1: Select Dataset]
          │
          ▼
[Stage 2: Hyperparameters & Advisor]
          │
          ▼
[Stage 3: Interactive Graph Explorer]
          │
          ▼
[Stage 4: Evaluation & Benchmarking]
          │
          ▼
[Stage 5: Results & Export]
```

### Stage 1: Dataset Selection and Ingestion
- Supports four dataset sources:
  1. **Synthetic Social Network:** Configurable Barabási-Albert / Watts-Strogatz small-world honest network (1,000 nodes default) attacked by a dense Sybil cluster across a sparse bottleneck cut.
  2. **Cora Citation Network:** 2,708 academic publication nodes and 10,556 citation edges (binary Sybil-adversary projection).
  3. **PubMed Citation Network:** 19,717 publication nodes and 88,648 edges for large-scale stress testing.
  4. **Custom CSV Upload:** Custom `edges.csv` and optional `labels.csv` files.
- Automatically normalizes node identifiers, deduplicates undirected edges, strips self-loops, and validates label consistency.

### Stage 2: Graph Advisor and Hyperparameter Optimization
- Automated heuristic analyzer inspects global topology: node count ($|V|$), edge count ($|E|$), density, connected components, isolated vertices, and labeled seed coverage.
- Computes recommended hyperparameter configurations:
  - **Propagation Factor ($\alpha$):** Balances neighbor diffusion vs. seed label retention ($0.5 \le \alpha \le 0.99$).
  - **Transition Operator:** Random-walk normalized Laplacian ($P = D^{-1}A$) or symmetric normalized Laplacian ($P = D^{-1/2}AD^{-1/2}$).
  - **Clamping Mode:** Hard clamping (pins training seed beliefs to ground truth every iteration) or Soft clamping (smooth regularization).
  - **Max Iterations & Convergence Tolerance ($\epsilon$):** Sets power-iteration termination criteria.

### Stage 3: Interactive Network Graph Explorer
- Interactive 2D force-directed physics network visualization powered by PyVis.
- Taxonomy-based color scheme:
  - Honest Seeds: Forest Green
  - Sybil Seeds: Crimson Red
  - Predicted Honest: Royal Blue
  - Predicted Sybil: Dark Orange
  - Unlabeled / Indeterminate: Slate Gray
- Performance-preserving subgraph filters to sustain 60 FPS physics rendering on graphs with up to tens of thousands of nodes:
  - Full sampled network view
  - Sybils and immediate boundary neighbors view
  - Ego-network 2-hop focal neighborhood view
- Real-time node inspector displays degree centrality, clustering coefficients, and neighbor class breakdowns.

### Stage 4: Leakage-Free Semi-Supervised Evaluation
- Strict split of available ground-truth labels into training seeds (20% default) and withheld test nodes (80% default).
- Withheld labels are completely shielded during the iterative label propagation phase to eliminate data snooping and test leakage.
- Reports comprehensive academic classification metrics:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - Specificity
  - F1-Score
  - Area Under the ROC Curve (ROC-AUC)
- Generates publication-grade Confusion Matrix heatmaps and ROC curves.
- Side-by-side benchmarking against scikit-learn's `LabelSpreading` baseline.

### Stage 5: Detection Results and Export
- Displays complete node-level prediction rosters with continuous belief values, binary class assignments, ground truth comparisons, and degree metrics.
- Provides downloadable CSV prediction records and serialized JSON benchmark reports for downstream forensic auditing.

---

## Experimental Benchmark Results

Evaluated on Apple Silicon M-series hardware using a standard 20% training seed and 80% strictly withheld test split:

| Dataset | Total Nodes | Total Edges | Algorithm | Accuracy | Precision | Recall | F1-Score | Iterations | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | 1,000 | 7,294 | **GraphShield (Sparse LP)** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **31** | **0.032s** |
| Synthetic | 1,000 | 7,294 | Baseline (LabelSpreading) | 100.0% | 1.0000 | 1.0000 | 1.0000 | 39 | 0.035s |
| **Cora** | 2,708 | 10,556 | **GraphShield (Sparse LP)** | **78.8%** | **0.7943** | **0.7798** | **0.7824** | **40** | **0.109s** |
| Cora | 2,708 | 10,556 | Baseline (LabelSpreading) | 79.7% | 0.8131 | 0.7852 | 0.7937 | 50 | 0.125s |
| **PubMed** | 19,717 | 88,648 | **GraphShield (Sparse LP)** | **81.1%** | **0.8045** | **0.7916** | **0.7968** | **41** | **0.888s** |
| PubMed | 19,717 | 88,648 | Baseline (LabelSpreading) | 81.9% | 0.8128 | 0.7995 | 0.8047 | 50 | 0.940s |

### Key Experimental Insights
1. **Bounded Attack Cut Detection:** On the synthetic network with a bounded quotient cut between honest and adversary regions, GraphShield achieves 100.0% detection accuracy and 1.0000 ROC-AUC, confirming the theoretical SybilGuard boundary condition.
2. **Sub-Second Scalability:** On the 19,717-node PubMed citation network, sparse power iterations complete in under 0.9 seconds (0.888s), verifying readiness for large online social networks.
3. **Equivalence to Dense Baselines:** GraphShield matches or exceeds scikit-learn's `LabelSpreading` across convergence speed and memory overhead while avoiding dense matrix materialization.

---

## Publication Artifacts and Output Directory (`outputs/`)

All publication-ready charts, serialized experimental benchmarks, and standalone reference implementations are cataloged in the `outputs/` folder:

### 1. High-Resolution Publication Figures (300 DPI, White Background)
All figures are rendered with crisp white backgrounds and publication typography for seamless inclusion in academic reports, LaTeX papers, and slide decks:

| File Name | Description |
| :--- | :--- |
| `01_confusion_matrix_synthetic.png` | Confusion matrix of held-out test accounts showing zero false positives or false negatives under bounded attack cut. |
| `02_roc_curve_synthetic.png` | Receiver Operating Characteristic (ROC) curve showing True Positive Rate vs. False Positive Rate (AUC = 1.0000). |
| `03_degree_distribution_synthetic.png` | Node connectivity degree distribution histogram across the 1,000-node synthetic benchmark network (mean degree = 14.6). |
| `04_benchmark_comparison_metrics.png` | Grouped comparative bar chart evaluating Accuracy and F1-Score across Synthetic, Cora, and PubMed networks. |
| `05_convergence_and_runtime.png` | Comparative bar plots of inference runtime and power iterations to convergence across all datasets. |
| `06_sybil_graph_topology.png` | 2D topological layout visualizing the Honest community, Sybil adversarial cell, and the sparse attack bridge cut. |

### 2. Experimental Data and Metrics
| File Name | Description | Format |
| :--- | :--- | :--- |
| `evaluation_summary.txt` | Formatted comparison table with exact Accuracy, Precision, Recall, F1, iterations, and runtimes across all benchmarks. | Plain Text |
| `benchmark_results.json` | Comprehensive machine-readable benchmark metrics, confusion matrices, and dataset dimensions. | JSON |
| `synthetic_predictions_sample.csv` | Sample predictions inventory (Node ID, Ground Truth, Predicted Class, Continuous Confidence Score, Node Degree). | CSV |

### 3. Standalone Reference Implementation
| File Name | Description |
| :--- | :--- |
| `graphshield_core_engine.py` | Self-contained, zero-dependency reference implementation of the sparse label propagation algorithm supporting Random Walk and Symmetric Laplacian transition operators. |

---

## Repository Architecture

```text
graph-sybil-detection/
├── app.py                      # Main Streamlit web dashboard application
├── requirements.txt            # Pinned macOS-compatible Python dependencies
├── README.md                   # Complete system documentation and macOS guide
├── .gitignore                  # Git ignore rules for virtualenvs, cache, and OS files
├── data/
│   ├── demo_edges.csv          # Pre-generated 1,000-node demo edge list
│   ├── demo_labels.csv         # Pre-generated seed labels for demo network
│   └── planetoid/              # Cached Cora and PubMed benchmark datasets (auto-downloaded)
├── outputs/                    # Publication figures, benchmark results, and artifacts
│   ├── 01_confusion_matrix_synthetic.png
│   ├── 02_roc_curve_synthetic.png
│   ├── 03_degree_distribution_synthetic.png
│   ├── 04_benchmark_comparison_metrics.png
│   ├── 05_convergence_and_runtime.png
│   ├── 06_sybil_graph_topology.png
│   ├── benchmark_results.json
│   ├── evaluation_summary.txt
│   ├── synthetic_predictions_sample.csv
│   ├── graphshield_core_engine.py
│   └── README.md
├── src/
│   ├── __init__.py             # Module initializer
│   ├── utils.py                # Hardware inspection (macOS, MPS, CPU) and seeding
│   ├── advisor.py              # Automatic dataset analyzer and heuristic recommendations
│   ├── data_loader.py          # CSV loading, schema validation, and graph statistics
│   ├── graph_builder.py        # NetworkX graph builder and sparse matrix converter
│   ├── synthetic_data.py       # Configurable social network and Sybil generator
│   ├── benchmark_datasets.py   # Cora and PubMed PyTorch Geometric loaders
│   ├── label_propagation.py    # Custom GraphShield LP and Sklearn baseline
│   ├── evaluation.py           # Leakage-free test evaluation and benchmark suite
│   └── visualization.py        # PyVis interactive network and Plotly diagnostic charts
├── tests/
│   ├── __init__.py             # Test suite initializer
│   ├── test_advisor.py         # Dataset advisor tests
│   ├── test_benchmark_datasets.py # Cora and PubMed loader tests
│   ├── test_data_loader.py     # CSV validation and graph stats tests
│   ├── test_graph_builder.py   # NetworkX builder and sparse matrix tests
│   ├── test_synthetic_data.py  # Synthetic generator and reproducibility tests
│   ├── test_label_propagation.py # Custom LP, clamping, and PyTorch MPS tests
│   └── test_evaluation.py      # Leakage prevention and metric evaluation tests
└── docs/
    ├── algorithm.md            # Mathematical formulation and convergence proof
    └── methodology.md          # Sybil defense literature and evaluation protocol
```

---

## macOS Setup Requirements

GraphShield has been developed and verified on macOS (compatible with both Apple Silicon `arm64` and Intel `x86_64`).

### Prerequisites

Ensure you have **Python 3.11** installed. If not already installed, install via [Homebrew](https://brew.sh):

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11
```

### Installation Steps

1. **Clone or Navigate to the Repository:**
   ```bash
   cd graph-sybil-detection
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Upgrade Pip and Install Dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Launch the GraphShield Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## Running Automated Tests

GraphShield includes an automated test suite verifying data validation, graph construction, dataset recommendations, benchmark loaders, label propagation correctness, Apple MPS backend compatibility, and leakage prevention:

```bash
# Run pytest with verbose test reporting
.venv/bin/python -m pytest -v
```

Expected output:
```text
============================== 23 passed in 2.36s ==============================
```

---

## Dataset CSV Format Specifications

GraphShield supports custom CSV uploads formatted as follows:

### 1. `edges.csv` (Required)
A header row with `source,target` columns:
```csv
source,target
alice,bob
bob,carol
carol,dave
```
*Notes:*
- Self-loops (`u == u`) are automatically detected and removed.
- Duplicate edges are deduplicated (canonicalized for undirected graphs).

### 2. `labels.csv` (Optional)
A header row with `node_id,label` columns:
```csv
node_id,label
alice,0
eve,1
bob,-1
```
*Label Conventions:*
- `0`: Known Normal / Honest account (training seed).
- `1`: Known Sybil / Malicious account (training seed).
- `-1`, blank, or omitted: Unlabeled account (will be inferred by the model).

---

## Mathematical Formulation

GraphShield implements discrete harmonic label propagation:

$$
Y^{(t+1)} = \alpha P Y^{(t)} + (1 - \alpha) Y^{(0)}
$$

where:
- $P = D^{-1} A$ is the row-stochastic random-walk transition matrix.
- $Y^{(0)} \in \mathbb{R}^{n \times K}$ is the initial label matrix ($[1, 0]$ for Normal, $[0, 1]$ for Sybil, $[0.5, 0.5]$ for unlabeled).
- $\alpha \in (0, 1)$ balances neighbor consensus with initial evidence.
- Under **Hard Clamping**, rows corresponding to labeled training seeds are clamped: $Y_L^{(t+1)} = Y_L^{(0)}$.
- Convergence is achieved when $\|Y^{(t+1)} - Y^{(t)}\|_F < \epsilon$.

For detailed convergence proofs, contraction mapping analysis, and complexity bounds, see [docs/algorithm.md](docs/algorithm.md).

For academic context regarding the bounded quotient cut assumption and attack topologies, see [docs/methodology.md](docs/methodology.md).

---

## macOS and Hardware Troubleshooting

| Scenario | Diagnosis / Solution |
|---|---|
| **MPS backend unavailable** | GraphShield automatically detects that `torch.backends.mps.is_available()` is `False` and falls back to CPU without crashing. |
| **Intel Mac Execution** | Runs seamlessly using NumPy/SciPy OpenBLAS CPU matrix routines without requiring MPS or CUDA. |
| **Port 8501 already in use** | Run with an alternate port: `streamlit run app.py --server.port 8502`. |
| **Large Graph Lag** | In the **Graph Explorer** tab, use the **View Filter** dropdown (`Sybils & Neighbors` or `Ego-Network`) or set `Max Display Nodes` to $\le 150$ to maintain 60 FPS physics simulation. |

---

## Academic Integrity and Citation

This software is developed for research and educational purposes in Graph Machine Learning and Social Network Security. Correlation in graph proximity does not constitute definitive proof of account compromise. Real-world fraud detection pipelines should combine graph-structural features with behavioral, content, and device fingerprint signals.
