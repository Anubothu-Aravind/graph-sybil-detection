# GraphShield: Graph-Based Sybil Node Detection System

[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon%20%26%20Intel)-brightgreen.svg)](#macos-setup-requirements)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](#installation)
[![Hardware: Apple Metal (MPS)](https://img.shields.io/badge/Hardware-Apple%20Metal%20MPS%20%2F%20CPU-orange.svg)](#apple-silicon-metal-performance-shaders-mps)
[![Tests: Passing](https://img.shields.io/badge/Tests-23%20Passed-success.svg)](#running-automated-tests)

**GraphShield** is an academic-grade Graph Machine Learning application that detects Sybil (fake/malicious) nodes in social networks using semi-supervised graph label propagation.

Designed specifically for **macOS** (both Apple Silicon and Intel architectures), GraphShield runs locally without requiring CUDA-enabled NVIDIA GPUs, taking advantage of **Apple Metal Performance Shaders (MPS)** unified memory acceleration with automatic CPU fallback.

---

## Key Features

1. **Intelligent Dataset Advisor:**
   - Automatically analyzes graph size, density, connectivity, isolated nodes, and label coverage upon selection.
   - Computes data-driven heuristic settings (recommended algorithm, propagation parameter, iteration limit, and convergence tolerance).

2. **One-Click Detection Workflow:**
   - Prominent single-click detection pipeline executing dataset validation, graph building, heuristic configuration, label propagation, evaluation, and interactive rendering.

3. **Multi-Dataset Benchmarking:**
   - **Dataset A (Default):** Configurable Synthetic Social Network (1,000 nodes, 5,000–10,000 edges) with realistic community structure and Sybil attack cut modeling.
   - **Dataset B:** Cora Academic Citation Network (2,708 nodes, 10,556 edges) loaded via PyTorch Geometric.
   - **Dataset C:** PubMed Publication Citation Network (19,717 nodes, 88,648 edges) for scalability experiments.
   - **Dataset D:** Custom CSV file upload (`edges.csv`, optional `labels.csv`).

4. **Apple Metal Performance Shaders (MPS) Acceleration:**
   - Detects host Mac architecture (Apple Silicon vs. Intel).
   - Utilizes Apple Metal GPU tensor acceleration via PyTorch MPS when available.
   - Automatically falls back to high-performance NumPy/SciPy sparse BLAS on CPU without crashing.

5. **Rigorous Evaluation without Test Leakage:**
   - Ground-truth evaluation withheld completely during propagation.
   - Reports Accuracy, Precision, Recall, F1-Score, Specificity, and ROC-AUC.
   - Interactive Confusion Matrix heatmaps and Receiver Operating Characteristic (ROC) curves.
   - Side-by-side benchmark comparison against scikit-learn's `LabelSpreading`.

6. **Interactive Network Visualization:**
   - Force-directed physics visualization powered by PyVis.
   - Distinct color and shape taxonomy (Known Normal, Known Sybil, Predicted Normal, Predicted Sybil, Unlabeled).
   - Responsive sampling and neighborhood filtering (Ego-network 2-hop view, Sybils + Neighbors) to maintain 60 FPS on large graphs.
   - Real-time node inspector showing degrees, clustering, and suspicious neighborhood counts.

---

## Repository Architecture

```text
graph-sybil-detection/
├── app.py                      # Main Streamlit web dashboard application
├── requirements.txt            # Pinned macOS-compatible Python dependencies
├── README.md                   # Complete system documentation and macOS guide
├── .gitignore                  # Git ignore rules for virtualenvs and OS files
├── data/
│   ├── demo_edges.csv          # Pre-generated 1,000-node demo edge list
│   ├── demo_labels.csv         # Pre-generated seed labels for demo network
│   └── planetoid/              # Cached Cora and PubMed benchmark datasets
├── src/
│   ├── __init__.py             # Module initializer
│   ├── utils.py                # Hardware inspection (macOS, MPS, CPU) & seeding
│   ├── advisor.py              # Automatic dataset analyzer & heuristic recommendations
│   ├── data_loader.py          # CSV loading, schema validation & graph statistics
│   ├── graph_builder.py        # NetworkX graph builder & sparse matrix converter
│   ├── synthetic_data.py       # Configurable social network & Sybil generator
│   ├── benchmark_datasets.py   # Cora and PubMed PyTorch Geometric loaders
│   ├── label_propagation.py    # Custom GraphShield LP & Sklearn baseline
│   ├── evaluation.py           # Leakage-free test evaluation & benchmark suite
│   └── visualization.py        # PyVis interactive network & Plotly diagnostic charts
├── tests/
│   ├── __init__.py             # Test suite initializer
│   ├── test_advisor.py         # Dataset advisor tests
│   ├── test_benchmark_datasets.py # Cora and PubMed loader tests
│   ├── test_data_loader.py     # CSV validation and graph stats tests
│   ├── test_graph_builder.py   # NetworkX builder and sparse matrix tests
│   ├── test_synthetic_data.py  # Synthetic generator & reproducibility tests
│   ├── test_label_propagation.py # Custom LP, clamping, & PyTorch MPS tests
│   └── test_evaluation.py      # Leakage prevention & metric evaluation tests
└── docs/
    ├── algorithm.md            # Mathematical formulation & convergence proof
    └── methodology.md          # Sybil defense literature & evaluation protocol
```

---

## macOS Setup Requirements

GraphShield has been developed and verified on macOS (compatible with both Apple Silicon `arm64` and Intel `x86_64`).

### Prerequisites

Ensure you have **Python 3.11** installed. If not already installed, you can install it using [Homebrew](https://brew.sh):

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
============================== 23 passed in 2.24s ==============================
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

## macOS & Hardware Troubleshooting

| Scenario | Diagnosis / Solution |
|---|---|
| **MPS backend unavailable** | GraphShield automatically detects that `torch.backends.mps.is_available()` is `False` and falls back to CPU without crashing. |
| **Intel Mac Execution** | Runs seamlessly using NumPy/SciPy OpenBLAS CPU matrix routines without requiring MPS or CUDA. |
| **Port 8501 already in use** | Run with an alternate port: `streamlit run app.py --server.port 8502`. |
| **Large Graph Lag** | In the **Graph Explorer** tab, use the **View Filter** dropdown (`Sybils & Neighbors` or `Ego-Network`) or set `Max Display Nodes` to $\le 150$ to maintain 60 FPS physics simulation. |

---

## Academic Integrity & Citation

This software is developed for research and educational purposes in Graph Machine Learning and Social Network Security. Correlation in graph proximity does not constitute definitive proof of account compromise. Real-world fraud detection pipelines should combine graph-structural features with behavioral, content, and device fingerprint signals.
