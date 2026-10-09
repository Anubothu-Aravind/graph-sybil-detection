"""GraphShield: Intelligent Graph-Based Sybil Detection System.

Streamlit application implementing a guided 5-step detection pipeline:
1. Select Dataset (Synthetic, Cora, PubMed, Custom CSV)
2. Hyperparameters (Advisor recommendations, algorithm parameters)
3. Interactive Graph Explorer (PyVis interactive graph, filters, inspector)
4. Evaluation (Leakage-free held-out evaluation, side-by-side benchmark)
5. Results (Calculations, predictions table, metrics, export)
"""

import os
import json
import io
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import streamlit.components.v1 as components

from src.utils import get_device_info, set_seed
from src.data_loader import (
    load_and_validate_edges,
    load_and_validate_labels,
    compute_graph_statistics,
    ValidationError
)
from src.synthetic_data import generate_synthetic_social_network, generate_demo_dataset
from src.benchmark_datasets import load_cora_dataset, load_pubmed_dataset
from src.advisor import analyze_and_recommend
from src.graph_builder import build_networkx_graph
from src.label_propagation import GraphShieldLabelPropagation, BaselineLabelSpreading
from src.evaluation import (
    split_train_test_labels,
    compute_classification_metrics,
    evaluate_detection,
    run_benchmark_comparison
)
from src.visualization import (
    build_interactive_graph_html,
    plot_degree_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_confidence_distribution,
    COLOR_MAP
)

# Page Configuration - Strictly No Emojis
st.set_page_config(
    page_title="GraphShield | Sybil Detection Pipeline",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark glassmorphic design system tokens
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: #f8fafc;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(30, 41, 59, 0.45) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(15, 23, 42, 0.6) 0%, transparent 45%),
                #090d16;
}

/* Glassmorphism Cards */
.gs-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

.gs-card-advisor {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px -4px rgba(14, 165, 233, 0.15);
}

/* Pipeline Stepper */
.pipeline-stepper {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 24px;
    overflow-x: auto;
}

.step-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: #94a3b8;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    white-space: nowrap;
}

.step-active {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.5) !important;
    font-weight: 600;
}

.step-completed {
    background: rgba(16, 185, 129, 0.12) !important;
    color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
}

.step-arrow {
    color: #475569;
    font-size: 13px;
    user-select: none;
}

/* Metric Display */
.metric-container {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-val {
    font-size: 26px;
    font-weight: 700;
    color: #f8fafc;
    font-feature-settings: "tnum";
}
.metric-lbl {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
}

.badge-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-mps {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.badge-cpu {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.badge-sybil {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.3);
}
.badge-normal {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}

div[data-testid="stSidebar"] {
    background-color: #0b1120;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    color: #94a3b8;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Hardware Platform Detection
device_info = get_device_info()

PIPELINE_STEPS = [
    "1. Select Dataset",
    "2. Hyperparameters",
    "3. Interactive Graph Explorer",
    "4. Evaluation",
    "5. Results"
]


def initialize_session_state():
    """Initialize session state parameters for the pipeline flow."""
    if "pipeline_step" not in st.session_state:
        st.session_state.pipeline_step = PIPELINE_STEPS[0]
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "current_dataset_name" not in st.session_state:
        st.session_state.current_dataset_name = "Synthetic Social Network"
    if "edges_df" not in st.session_state:
        st.session_state.edges_df = None
    if "labels_df" not in st.session_state:
        st.session_state.labels_df = None
    if "ground_truth" not in st.session_state:
        st.session_state.ground_truth = None
    if "dataset_meta" not in st.session_state:
        st.session_state.dataset_meta = {}
    if "G" not in st.session_state:
        st.session_state.G = None
    if "graph_stats" not in st.session_state:
        st.session_state.graph_stats = None
    if "advisor_report" not in st.session_state:
        st.session_state.advisor_report = None
    if "model_results" not in st.session_state:
        st.session_state.model_results = None
    if "eval_metrics" not in st.session_state:
        st.session_state.eval_metrics = None
    if "benchmark_df" not in st.session_state:
        st.session_state.benchmark_df = None
    if "run_history" not in st.session_state:
        st.session_state.run_history = []


initialize_session_state()


def set_dataset(name: str, edges_df: pd.DataFrame, labels_df: pd.DataFrame, gt_dict: dict, meta: dict):
    """Set active dataset and trigger automatic structural advisor analysis."""
    G, _ = build_networkx_graph(edges_df, labels_df)
    stats = compute_graph_statistics(G, labels_df)

    known_labels = {n: G.nodes[n].get("label", -1) for n in G.nodes()}
    advisor_report = analyze_and_recommend(G, known_labels)

    st.session_state.current_dataset_name = name
    st.session_state.edges_df = edges_df
    st.session_state.labels_df = labels_df
    st.session_state.ground_truth = gt_dict
    st.session_state.dataset_meta = meta
    st.session_state.G = G
    st.session_state.graph_stats = stats
    st.session_state.advisor_report = advisor_report
    st.session_state.model_results = None
    st.session_state.eval_metrics = None
    st.session_state.benchmark_df = None
    st.session_state.data_loaded = True


def load_default_synthetic_dataset():
    """Load default 1,000-node synthetic social network."""
    demo_edges = "data/demo_edges.csv"
    demo_labels = "data/demo_labels.csv"

    if not os.path.exists(demo_edges) or not os.path.exists(demo_labels):
        generate_demo_dataset("data")

    edges_df, _ = load_and_validate_edges(demo_edges)
    labels_df, _ = load_and_validate_labels(demo_labels)

    gt = {}
    for _, row in labels_df.iterrows():
        nid = str(row["node_id"])
        if nid.startswith("S_"):
            gt[nid] = 1
        elif nid.startswith("N_"):
            gt[nid] = 0
        else:
            gt[nid] = int(row["label"]) if int(row["label"]) in [0, 1] else 0

    meta = {
        "dataset_name": "Synthetic Social Network",
        "dataset_key": "synthetic",
        "is_synthetic": True,
        "is_citation_benchmark": False,
        "num_total_nodes": len(labels_df),
        "num_edges": len(edges_df),
        "notice": "Simulated synthetic social network with realistic community structure and a bounded Sybil region."
    }

    set_dataset("Synthetic Social Network", edges_df, labels_df, gt, meta)


# Auto-load synthetic demo network on first initialization
if not st.session_state.data_loaded:
    load_default_synthetic_dataset()


def execute_detection_pipeline(cfg: dict) -> dict:
    """Execute label propagation detection and record results."""
    G = st.session_state.G
    known_labels = {n: G.nodes[n].get("label", -1) for n in G.nodes()}
    gt = st.session_state.ground_truth or {}

    start_time = time.perf_counter()
    set_seed(cfg.get("seed", 42))

    use_torch = "PyTorch" in cfg["algorithm"] or "MPS" in cfg["algorithm"]
    device = "mps" if (use_torch and device_info["mps_available"]) else "cpu"

    if "Baseline" in cfg["algorithm"]:
        model = BaselineLabelSpreading(
            alpha=cfg["alpha"],
            max_iter=cfg["max_iter"],
            tol=cfg["tol"]
        )
    else:
        model = GraphShieldLabelPropagation(
            alpha=cfg["alpha"],
            max_iter=cfg["max_iter"],
            tol=cfg["tol"],
            clamping=cfg.get("clamping", True),
            normalization=cfg.get("normalization", "random_walk"),
            threshold=cfg.get("threshold", 0.50),
            use_torch=use_torch,
            device=device
        )

    results = model.fit_predict(G, known_labels)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    results["elapsed_ms"] = round(elapsed_ms, 2)
    results["algorithm_used"] = cfg["algorithm"]

    eval_metrics = None
    if gt and len(gt) > 0:
        is_citation = st.session_state.dataset_meta.get("is_citation_benchmark", False)
        valid_gt_nodes = [n for n in gt.keys() if n in G]

        # Evaluate strictly against withheld test nodes to ensure zero leakage
        test_nodes = [n for n in valid_gt_nodes if known_labels.get(n, -1) == -1]

        if len(test_nodes) > 0:
            y_true = [gt[n] for n in test_nodes]
            y_pred = [results["predictions"][n] for n in test_nodes]
            y_scores = [results["confidence_scores"][n] for n in test_nodes]
            eval_metrics = compute_classification_metrics(y_true, y_pred, y_scores)
            eval_metrics["test_nodes_count"] = len(test_nodes)
            eval_metrics["is_citation_benchmark"] = is_citation

    run_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "dataset": str(st.session_state.current_dataset_name),
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "predicted_sybils": int(sum(1 for p in results["predictions"].values() if p == 1)),
        "runtime_ms": float(results["elapsed_ms"]),
        "accuracy": round(float(eval_metrics["accuracy"]), 4) if eval_metrics and eval_metrics.get("accuracy") is not None else None,
        "f1_score": round(float(eval_metrics["f1_score"]), 4) if eval_metrics and eval_metrics.get("f1_score") is not None else None,
    }
    st.session_state.run_history.insert(0, run_entry)
    st.session_state.run_history = st.session_state.run_history[:5]

    st.session_state.model_results = results
    st.session_state.eval_metrics = eval_metrics

    return results


def render_pipeline_stepper(current_step: str):
    """Render top glassmorphic pipeline progress indicator."""
    steps = [
        ("1. Select Dataset", "1. Dataset"),
        ("2. Hyperparameters", "2. Hyperparameters"),
        ("3. Interactive Graph Explorer", "3. Graph Explorer"),
        ("4. Evaluation", "4. Evaluation"),
        ("5. Results", "5. Results")
    ]
    cur_idx = 0
    for i, (full_name, _) in enumerate(steps):
        if full_name == current_step:
            cur_idx = i
            break

    stepper_html = '<div class="pipeline-stepper">'
    for i, (full_name, label) in enumerate(steps):
        cls = "step-item"
        if i == cur_idx:
            cls += " step-active"
        elif i < cur_idx:
            cls += " step-completed"

        marker = "[Done] " if i < cur_idx else f"[{i+1}] "
        stepper_html += f'<div class="{cls}">{marker}{label}</div>'
        if i < len(steps) - 1:
            stepper_html += '<div class="step-arrow">&rarr;</div>'
    stepper_html += '</div>'
    st.markdown(stepper_html, unsafe_allow_html=True)


# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("### **GraphShield**")
    st.caption("Intelligent Graph Sybil Detection Pipeline")

    st.markdown("---")
    st.markdown("#### **System & Acceleration**")
    if device_info["is_apple_silicon"]:
        st.markdown(f"**Platform:** Apple Silicon ({device_info['architecture'].upper()})")
    else:
        st.markdown(f"**Platform:** {device_info['os']} ({device_info['architecture']})")

    if device_info["mps_available"]:
        st.markdown('<span class="badge-pill badge-mps">[MPS Acceleration Active]</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-pill badge-cpu">[CPU Execution Mode]</span>', unsafe_allow_html=True)

    st.caption(device_info["details"])

    st.markdown("---")
    st.markdown("#### **Active Dataset**")
    st.markdown(f"**{st.session_state.current_dataset_name}**")
    if st.session_state.G:
        st.caption(f"{st.session_state.G.number_of_nodes():,} nodes | {st.session_state.G.number_of_edges():,} edges")

    st.markdown("---")
    st.markdown("#### **Pipeline Stages**")
    step_idx = PIPELINE_STEPS.index(st.session_state.pipeline_step) if st.session_state.pipeline_step in PIPELINE_STEPS else 0
    selected_step = st.radio(
        "Pipeline Stages",
        PIPELINE_STEPS,
        index=step_idx,
        label_visibility="collapsed"
    )
    if selected_step != st.session_state.pipeline_step:
        st.session_state.pipeline_step = selected_step
        st.rerun()

    st.markdown("---")
    if st.button("Reset Demo Dataset", width="stretch"):
        load_default_synthetic_dataset()
        st.session_state.pipeline_step = "1. Select Dataset"
        st.success("Loaded synthetic network.")
        st.rerun()

# =========================================================================
# TOP PIPELINE STEPPER
# =========================================================================
render_pipeline_stepper(st.session_state.pipeline_step)

# =========================================================================
# STEP 1: SELECT DATASET
# =========================================================================
if st.session_state.pipeline_step == "1. Select Dataset":
    st.markdown("## Step 1: Select Dataset & Graph Analysis")
    st.markdown(
        "Choose a benchmark social network, generate a configurable synthetic graph, or upload your own edge list. "
        "GraphShield automatically computes graph statistics and diagnoses structural connectivity."
    )

    t_synth, t_cora, t_pubmed, t_upload = st.tabs([
        "Dataset A: Synthetic Network (1,000 nodes)",
        "Dataset B: Cora Citation (2,708 nodes)",
        "Dataset C: PubMed Citation (19,717 nodes)",
        "Dataset D: Custom CSV Upload"
    ])

    # Tab A: Synthetic
    with t_synth:
        st.markdown("#### Configurable Synthetic Social Network")
        st.markdown(
            "Generates a multi-community social network with normal users and a coordinated Sybil region. "
            "Includes separate ground-truth labels for evaluation that are withheld from model predictions."
        )

        c1, c2 = st.columns(2)
        with c1:
            s_nodes = st.slider("Total Nodes", 200, 2000, 1000, 100)
            s_sybil_ratio = st.slider("Sybil Ratio", 0.05, 0.40, 0.20, 0.05)
            s_topology = st.selectbox("Sybil Attack Topology", ["clustered", "star", "random"])
        with c2:
            s_labeled_ratio = st.slider("Known Seed Label Percentage", 0.05, 0.40, 0.20, 0.05)
            s_attack_edges = st.slider("Attack Edges (Quotient Cut)", 4, 30, 12, 2)
            s_seed = st.number_input("Random Seed", 1, 99999, 42)

        if st.button("Generate & Load Synthetic Network", type="primary", width="stretch"):
            with st.spinner("Synthesizing network topology..."):
                e_df, l_df, gt_df, meta = generate_synthetic_social_network(
                    num_nodes=s_nodes,
                    sybil_ratio=s_sybil_ratio,
                    labeled_ratio=s_labeled_ratio,
                    sybil_topology=s_topology,
                    attack_edges_count=s_attack_edges,
                    seed=s_seed
                )
                gt = {str(row["node_id"]): int(row["ground_truth"]) for _, row in gt_df.iterrows()}
                set_dataset("Synthetic Social Network", e_df, l_df, gt, meta)
                st.success(f"Generated synthetic network with {s_nodes:,} nodes and {len(e_df):,} edges.")
                st.rerun()

    # Tab B: Cora
    with t_cora:
        st.markdown("#### Cora Academic Citation Network")
        st.markdown(
            "The Cora dataset consists of 2,708 scientific publications classified into 7 research topics, "
            "connected by 10,556 citation edges. Loaded via PyTorch Geometric."
        )
        st.info(
            "Academic Disclosure: Cora is an academic citation network used to demonstrate graph scalability "
            "and semi-supervised classification. Topic classes represent scientific domains, not malicious accounts."
        )
        if st.button("Load Cora Citation Network", type="primary", width="stretch"):
            with st.spinner("Loading Cora dataset via PyTorch Geometric..."):
                try:
                    e_df, l_df, gt_df, meta = load_cora_dataset("data/planetoid")
                    gt = {str(row["node_id"]): int(row["ground_truth"]) for _, row in gt_df.iterrows()}
                    set_dataset("Cora Citation Network", e_df, l_df, gt, meta)
                    st.success("Loaded Cora Citation Network (2,708 nodes, 10,556 edges).")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error loading Cora: {ex}")

    # Tab C: PubMed
    with t_pubmed:
        st.markdown("#### PubMed Publication Citation Network")
        st.markdown(
            "The PubMed dataset contains 19,717 scientific publications concerning diabetes, categorized into "
            "3 classes, connected by 88,648 citation edges. Demonstrates large-scale sparse propagation."
        )
        st.info(
            "Academic Disclosure: PubMed is an academic citation network used for scalability and general "
            "semi-supervised benchmarking, not as ground-truth Sybil data."
        )
        if st.button("Load PubMed Citation Network", type="primary", width="stretch"):
            with st.spinner("Loading PubMed dataset via PyTorch Geometric..."):
                try:
                    e_df, l_df, gt_df, meta = load_pubmed_dataset("data/planetoid")
                    gt = {str(row["node_id"]): int(row["ground_truth"]) for _, row in gt_df.iterrows()}
                    set_dataset("PubMed Citation Network", e_df, l_df, gt, meta)
                    st.success("Loaded PubMed Citation Network (19,717 nodes, 88,648 edges).")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error loading PubMed: {ex}")

    # Tab D: Custom Upload
    with t_upload:
        st.markdown("#### Custom CSV Upload")
        st.markdown("Upload your own edge list and optional seed label CSV files:")

        up_c1, up_c2 = st.columns(2)
        with up_c1:
            edges_file = st.file_uploader("Edges CSV (required)", type=["csv"], key="custom_edges_uploader")
            st.caption("Format: `source,target`")
        with up_c2:
            labels_file = st.file_uploader("Labels CSV (optional)", type=["csv"], key="custom_labels_uploader")
            st.caption("Format: `node_id,label` (0 = Normal, 1 = Sybil, blank = Unlabeled)")

        directed_opt = st.checkbox("Directed Graph", value=False)

        if st.button("Validate & Load Custom Dataset", type="primary", width="stretch"):
            if edges_file is None:
                st.error("Please upload an edges CSV file.")
            else:
                try:
                    e_df, edge_rep = load_and_validate_edges(edges_file, directed=directed_opt)
                    l_df = None
                    if labels_file is not None:
                        l_df, lbl_rep = load_and_validate_labels(labels_file)

                    gt = {}
                    if l_df is not None:
                        for _, row in l_df.iterrows():
                            if int(row["label"]) in [0, 1]:
                                gt[str(row["node_id"])] = int(row["label"])

                    meta = {
                        "dataset_name": "Custom Uploaded Dataset",
                        "dataset_key": "custom",
                        "is_synthetic": False,
                        "is_citation_benchmark": False,
                        "num_total_nodes": len(e_df),
                        "num_edges": len(e_df),
                        "notice": "User-uploaded dataset."
                    }

                    set_dataset("Custom Uploaded Dataset", e_df, l_df, gt, meta)
                    st.success(f"Custom dataset validated: {e_df.shape[0]} edges loaded.")
                    st.rerun()
                except ValidationError as ve:
                    st.error(f"Validation Error: {ve}")
                except Exception as ex:
                    st.error(f"Error reading CSV: {ex}")

    # Graph Statistics & Structural Diagnostics
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Graph Summary & Advisor Diagnostics")

    stats = st.session_state.graph_stats or {}
    adv = st.session_state.advisor_report or {}

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Total Nodes</div>
        <div class="metric-val">{stats.get('num_nodes', 0):,}</div>
    </div>
    """, unsafe_allow_html=True)
    kpi2.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Total Edges</div>
        <div class="metric-val">{stats.get('num_edges', 0):,}</div>
    </div>
    """, unsafe_allow_html=True)
    kpi3.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Avg Degree</div>
        <div class="metric-val">{stats.get('avg_degree', 0.0):.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    kpi4.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Components</div>
        <div class="metric-val">{stats.get('num_components', 0):,}</div>
    </div>
    """, unsafe_allow_html=True)
    kpi5.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Seed Labels</div>
        <div class="metric-val" style="color: #38bdf8;">{stats.get('labeled_count', 0):,}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    if st.session_state.G is not None:
        c_left, c_right = st.columns([1.2, 1])
        with c_left:
            fig_deg = plot_degree_distribution(st.session_state.G)
            st.plotly_chart(fig_deg, width="stretch")

        with c_right:
            summary_items = [
                {"Metric": "Dataset", "Value": str(st.session_state.current_dataset_name)},
                {"Metric": "Total Nodes", "Value": f"{stats.get('num_nodes', 0):,}"},
                {"Metric": "Total Edges", "Value": f"{stats.get('num_edges', 0):,}"},
                {"Metric": "Average Degree", "Value": f"{stats.get('avg_degree', 0.0):.2f}"},
                {"Metric": "Connected Components", "Value": f"{stats.get('num_components', 0):,}"},
                {"Metric": "Isolated Nodes", "Value": f"{stats.get('isolated_nodes', 0):,}"},
                {"Metric": "Largest Component Size", "Value": f"{stats.get('largest_component_size', 0):,}"},
                {"Metric": "Known Seed Labels", "Value": f"{stats.get('labeled_count', 0):,}"},
            ]
            st.dataframe(pd.DataFrame(summary_items), hide_index=True, width="stretch")

    # Advisor Structural Assessment
    if adv:
        st.markdown(f"""
        <div class="gs-card-advisor">
            <h4 style="margin-top:0; color: #38bdf8;">Automatic Graph Structural Analysis</h4>
            <p style="font-size: 13px; color: #cbd5e1; margin-bottom: 8px;">
                <b>Graph Scale:</b> {adv.get('size_category', 'Medium').capitalize()} ({stats.get('num_nodes', 0):,} nodes) &nbsp;|&nbsp;
                <b>Connectivity:</b> {adv.get('connectivity_status', 'Connected')} &nbsp;|&nbsp;
                <b>Seed Coverage:</b> {adv.get('coverage_status', 'Sufficient')} ({stats.get('labeled_count', 0):,} labeled nodes)
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline Forward CTA
    st.markdown("---")
    c_space, c_next = st.columns([2.5, 1.5])
    with c_next:
        if st.button("Continue to Hyperparameters ->", type="primary", width="stretch"):
            st.session_state.pipeline_step = "2. Hyperparameters"
            st.rerun()

# =========================================================================
# STEP 2: HYPERPARAMETERS
# =========================================================================
elif st.session_state.pipeline_step == "2. Hyperparameters":
    st.markdown("## Step 2: Hyperparameters & Algorithm Configuration")
    st.markdown(
        "Configure graph label propagation diffusion parameters or use recommended settings "
        "tailored to the active dataset by the structural analyzer."
    )

    if st.session_state.G is None:
        st.warning("Please select a dataset in Step 1 first.")
        st.stop()

    adv = st.session_state.advisor_report or {}
    adv_recs = adv.get("recommendations", {})

    st.markdown(f"""
    <div class="gs-card-advisor">
        <h4 style="margin-top:0; color: #38bdf8;">Recommended Settings for {st.session_state.current_dataset_name}</h4>
        <p style="font-size: 13px; color: #cbd5e1; margin-bottom: 8px;">
            Based on graph topology ({adv.get('size_category', 'Medium')}, {adv.get('connectivity_status', 'Connected')}),
            the following hyperparameters provide optimal convergence and boundary separation:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; font-size: 13px;">
            <div><b>Alpha (Diffusion):</b> {adv_recs.get('alpha', 0.85)}</div>
            <div><b>Max Iterations:</b> {adv_recs.get('max_iter', 100)}</div>
            <div><b>Convergence Tol:</b> {adv_recs.get('tol', 0.001)}</div>
            <div><b>Decision Threshold:</b> {adv_recs.get('threshold', 0.50)}</div>
            <div><b>Normalization:</b> Random Walk (D^-1 A)</div>
            <div><b>Clamping:</b> Hard Prior Clamping</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Configure Parameters")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        sel_algo = st.selectbox(
            "Algorithm Backend",
            [
                "GraphShield (Custom LP - NumPy/SciPy CPU)",
                "GraphShield (Apple MPS / PyTorch Accelerated)",
                "Baseline (scikit-learn LabelSpreading)"
            ],
            index=1 if device_info["mps_available"] else 0,
            help="Select PyTorch MPS acceleration for GPU hardware acceleration on Apple Silicon."
        )
        sel_alpha = st.slider(
            "Propagation Factor (alpha)",
            0.10, 0.99,
            float(adv_recs.get("alpha", 0.85)),
            0.05,
            help="Higher values allow trust signals to diffuse farther through the network."
        )
        sel_max_iter = st.slider(
            "Maximum Iterations",
            10, 500,
            int(adv_recs.get("max_iter", 100)),
            10,
            help="Maximum power iterations before stopping."
        )
        sel_tol = st.select_slider(
            "Convergence Tolerance",
            options=[0.01, 0.001, 0.0001, 0.00001],
            value=float(adv_recs.get("tol", 0.001)),
            help="Stopping threshold based on Frobenius norm between successive iterations."
        )

    with col_p2:
        sel_thresh = st.slider(
            "Sybil Decision Threshold",
            0.10, 0.90,
            float(adv_recs.get("threshold", 0.50)),
            0.05,
            help="Nodes with posterior Sybil confidence >= threshold are classified as Sybils."
        )
        sel_norm = st.selectbox(
            "Transition Normalization",
            ["random_walk", "symmetric"],
            index=0,
            format_func=lambda x: "Random Walk (D^-1 A)" if x == "random_walk" else "Symmetric Normalized (D^-1/2 A D^-1/2)"
        )
        sel_clamping = st.checkbox(
            "Hard Label Clamping (Keep training seeds fixed at each iteration)",
            value=True,
            help="Preserves known verified ground truth priors throughout the diffusion process."
        )
        sel_seed = st.number_input("Random Seed", 1, 99999, 42)

    active_cfg = {
        "algorithm": sel_algo,
        "alpha": sel_alpha,
        "max_iter": sel_max_iter,
        "tol": sel_tol,
        "threshold": sel_thresh,
        "clamping": sel_clamping,
        "normalization": sel_norm,
        "seed": sel_seed
    }

    # Pipeline CTAs
    st.markdown("---")
    c_back, c_run = st.columns([1, 2])
    with c_back:
        if st.button("<- Back to Dataset", width="stretch"):
            st.session_state.pipeline_step = "1. Select Dataset"
            st.rerun()
    with c_run:
        if st.button("Run Sybil Detection & Open Graph Explorer ->", type="primary", width="stretch"):
            with st.spinner("Executing label propagation pipeline..."):
                execute_detection_pipeline(active_cfg)
                st.session_state.pipeline_step = "3. Interactive Graph Explorer"
                st.rerun()

# =========================================================================
# STEP 3: INTERACTIVE GRAPH EXPLORER
# =========================================================================
elif st.session_state.pipeline_step == "3. Interactive Graph Explorer":
    st.markdown("## Step 3: Interactive Graph Explorer")
    st.markdown(
        "Inspect the full network topology, interact with force-directed physics, zoom into clusters, "
        "and observe how Sybil nodes coordinate along the attack boundary."
    )

    if st.session_state.G is None:
        st.warning("Please select a dataset in Step 1 first.")
        st.stop()

    # Automatically execute detection if not yet run
    if st.session_state.model_results is None:
        with st.spinner("Executing label propagation detection..."):
            adv_recs = st.session_state.advisor_report.get("recommendations", {}) if st.session_state.advisor_report else {}
            cfg = {
                "algorithm": adv_recs.get("algorithm", "GraphShield (Custom LP - NumPy/SciPy)"),
                "alpha": adv_recs.get("alpha", 0.85),
                "max_iter": adv_recs.get("max_iter", 100),
                "tol": adv_recs.get("tol", 0.001),
                "threshold": adv_recs.get("threshold", 0.50),
                "clamping": adv_recs.get("clamping", True),
                "normalization": adv_recs.get("normalization", "random_walk"),
                "seed": 42
            }
            execute_detection_pipeline(cfg)

    res = st.session_state.model_results
    known_labels = {n: st.session_state.G.nodes[n].get("label", -1) for n in st.session_state.G.nodes()}

    # Graph Controls Row
    gc1, gc2, gc3 = st.columns([1.5, 1.2, 1.3])
    with gc1:
        vis_filter = st.selectbox(
            "Subgraph View Mode",
            ["all", "sybils_and_neighbors", "ego", "suspicious_only"],
            format_func=lambda x: {
                "all": "Entire Graph (Sampled if large)",
                "sybils_and_neighbors": "Sybils & Immediate Neighbors",
                "ego": "Ego-Network of Focus Node (2-Hop)",
                "suspicious_only": "Only Flagged & Known Sybils"
            }[x]
        )
    with gc2:
        all_nodes = sorted(list(st.session_state.G.nodes()))
        focus_node = st.selectbox("Inspect Focus Node", options=all_nodes, index=0)
    with gc3:
        n_nodes_total = st.session_state.G.number_of_nodes()
        slider_max = max(50, min(1000, n_nodes_total))
        default_val = min(250, slider_max)
        max_vis = st.slider(
            "Max Display Nodes",
            min_value=30,
            max_value=slider_max,
            value=default_val,
            step=10,
            help="Limits rendered nodes in the interactive browser canvas to maintain smooth 60 FPS physics."
        )

    # Visual Taxonomy Legend
    st.markdown("""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; font-size: 13px;">
        <span style="display:flex; align-items:center; gap:6px;"><span style="width:12px; height:12px; border-radius:50%; background:#10B981; display:inline-block;"></span> <b>Known Normal Seed</b></span>
        <span style="display:flex; align-items:center; gap:6px;"><span style="width:12px; height:12px; border-radius:50%; background:#EF4444; display:inline-block;"></span> <b>Known Sybil Seed</b></span>
        <span style="display:flex; align-items:center; gap:6px;"><span style="width:12px; height:12px; border-radius:50%; background:#38BDF8; display:inline-block;"></span> <b>Predicted Normal</b></span>
        <span style="display:flex; align-items:center; gap:6px;"><span style="width:12px; height:12px; border-radius:50%; background:#F59E0B; display:inline-block;"></span> <b>Predicted Sybil</b></span>
        <span style="display:flex; align-items:center; gap:6px;"><span style="width:12px; height:12px; border-radius:50%; background:#64748B; display:inline-block;"></span> <b>Unlabeled Account</b></span>
    </div>
    """, unsafe_allow_html=True)

    html_code, stats_vis = build_interactive_graph_html(
        st.session_state.G,
        known_labels=known_labels,
        predictions=res["predictions"],
        confidence_scores=res["confidence_scores"],
        filter_mode=vis_filter,
        focus_node=focus_node,
        max_nodes=max_vis,
        height="640px"
    )

    if stats_vis["sampled"]:
        st.info(f"Rendering {stats_vis['nodes_displayed']} of {stats_vis['total_graph_nodes']} nodes to ensure responsive animation.")

    if hasattr(st, "iframe"):
        st.iframe(html_code, height=650)
    else:
        components.html(html_code, height=650, scrolling=False)

    # Node Inspector Card
    if focus_node and focus_node in st.session_state.G:
        f_pred = res["predictions"].get(focus_node, 0)
        f_conf = res["confidence_scores"].get(focus_node, 0.0)
        f_deg = st.session_state.G.degree(focus_node)
        f_known = known_labels.get(focus_node, -1)
        f_nbrs = list(st.session_state.G.neighbors(focus_node))
        f_sus_nbrs = sum(1 for n in f_nbrs if known_labels.get(n) == 1)

        st.markdown(f"""
        <div class="gs-card" style="margin-top: 16px;">
            <h4 style="margin-top:0; color: #38bdf8;">Node Inspector: {focus_node}</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; font-size: 13px;">
                <div><b>Predicted Class:</b> {'Sybil (Malicious)' if f_pred == 1 else 'Normal (Benign)'}</div>
                <div><b>Posterior Score:</b> {f_conf:.4f}</div>
                <div><b>Degree (Connections):</b> {f_deg}</div>
                <div><b>Prior Status:</b> {'Known Normal Seed' if f_known == 0 else ('Known Sybil Seed' if f_known == 1 else 'Unlabeled Prior')}</div>
                <div><b>Suspicious Neighbors:</b> {f_sus_nbrs} of {f_deg}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline CTAs
    st.markdown("---")
    c_back3, c_next3 = st.columns([1, 2])
    with c_back3:
        if st.button("<- Back to Hyperparameters", width="stretch"):
            st.session_state.pipeline_step = "2. Hyperparameters"
            st.rerun()
    with c_next3:
        if st.button("Continue to Evaluation ->", type="primary", width="stretch"):
            st.session_state.pipeline_step = "4. Evaluation"
            st.rerun()

# =========================================================================
# STEP 4: EVALUATION
# =========================================================================
elif st.session_state.pipeline_step == "4. Evaluation":
    st.markdown("## Step 4: Leakage-Free Held-Out Benchmark")
    st.markdown(
        "Evaluate semi-supervised detection accuracy against withheld test nodes that were masked during training. "
        "Neither test labels nor test attributes were exposed during label propagation, ensuring zero test leakage."
    )

    if st.session_state.G is None:
        st.warning("Please select a dataset in Step 1 first.")
        st.stop()

    gt = st.session_state.ground_truth or {}
    is_citation = st.session_state.dataset_meta.get("is_citation_benchmark", False)

    if not gt or len(gt) == 0:
        st.info("No ground-truth labels available for this dataset to compute held-out accuracy.")
        st.stop()

    if is_citation:
        st.info(
            f"Academic Benchmark Note: Evaluating topic classification on {st.session_state.current_dataset_name}. "
            "These metrics evaluate semi-supervised node classification across paper categories and do NOT represent "
            "Sybil account detection accuracy."
        )

    ev_col1, ev_col2 = st.columns([1.5, 2.5])
    with ev_col1:
        split_ratio = st.slider("Training Seed Label Ratio", 0.05, 0.50, 0.20, 0.05)

    if st.button("Run Side-by-Side Benchmark", type="primary", width="stretch") or st.session_state.benchmark_df is None:
        with st.spinner("Executing held-out evaluation against Baseline..."):
            custom_m, baseline_m, df_comp = run_benchmark_comparison(
                st.session_state.G,
                gt,
                train_ratio=split_ratio,
                alpha=0.85,
                max_iter=100,
                tol=0.001,
                clamping=True,
                threshold=0.50,
                seed=42
            )
            st.session_state.eval_metrics = custom_m
            st.session_state.baseline_metrics = baseline_m
            st.session_state.benchmark_df = df_comp

    if st.session_state.eval_metrics:
        m = st.session_state.eval_metrics

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
        c2.metric("Precision", f"{m['precision'] * 100:.1f}%")
        c3.metric("Recall", f"{m['recall'] * 100:.1f}%")
        c4.metric("F1-Score", f"{m['f1_score']:.3f}")

        st.markdown("<br/>", unsafe_allow_html=True)

        if m.get("is_binary"):
            ch1, ch2 = st.columns(2)
            with ch1:
                fig_cm = plot_confusion_matrix(m["confusion_matrix"])
                st.plotly_chart(fig_cm, width="stretch")
            with ch2:
                fig_roc = plot_roc_curve(m["fpr"], m["tpr"], m.get("roc_auc", 0.5))
                st.plotly_chart(fig_roc, width="stretch")

        st.markdown("### Side-by-Side Algorithm Comparison")
        if st.session_state.benchmark_df is not None:
            st.dataframe(st.session_state.benchmark_df, hide_index=True, width="stretch")

    # Pipeline CTAs
    st.markdown("---")
    c_back4, c_next4 = st.columns([1, 2])
    with c_back4:
        if st.button("<- Back to Graph Explorer", width="stretch"):
            st.session_state.pipeline_step = "3. Interactive Graph Explorer"
            st.rerun()
    with c_next4:
        if st.button("Proceed to Results & Calculations ->", type="primary", width="stretch"):
            st.session_state.pipeline_step = "5. Results"
            st.rerun()

# =========================================================================
# STEP 5: RESULTS
# =========================================================================
elif st.session_state.pipeline_step == "5. Results":
    st.markdown("## Step 5: Detection Results, Calculations & Export")
    st.markdown(
        "Review detection calculations, explore the complete account predictions inventory, "
        "and export results to CSV and JSON formats."
    )

    if st.session_state.model_results is None:
        st.warning("No detection run has been executed yet. Please run detection in Step 2.")
        st.stop()

    res = st.session_state.model_results
    gt = st.session_state.ground_truth or {}
    G = st.session_state.G

    preds = res["predictions"]
    confs = res["confidence_scores"]

    num_total = len(preds)
    num_sybils = sum(1 for p in preds.values() if p == 1)
    num_normal = num_total - num_sybils
    sybil_pct = (num_sybils / num_total * 100) if num_total > 0 else 0.0

    # Executive Calculations & Breakdown
    st.markdown("### Executive Summary & Calculations")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Total Accounts</div>
        <div class="metric-val">{num_total:,}</div>
    </div>
    """, unsafe_allow_html=True)
    r2.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Sybils Detected</div>
        <div class="metric-val" style="color: #ef4444;">{num_sybils:,}</div>
    </div>
    """, unsafe_allow_html=True)
    r3.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Normal Accounts</div>
        <div class="metric-val" style="color: #10b981;">{num_normal:,}</div>
    </div>
    """, unsafe_allow_html=True)
    r4.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Flagged Ratio</div>
        <div class="metric-val">{sybil_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    r5.markdown(f"""
    <div class="metric-container">
        <div class="metric-lbl">Runtime</div>
        <div class="metric-val" style="color: #38bdf8;">{res.get('elapsed_ms', 0):.1f}ms</div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Account Predictions Table
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Account Predictions Inventory")

    tf1, tf2, tf3 = st.columns([1.5, 1.5, 2])
    with tf1:
        filter_pred = st.selectbox("Filter by Classification", ["All Accounts", "Sybils Only", "Normal Only"])
    with tf2:
        min_conf = st.slider("Minimum Confidence Score", 0.0, 1.0, 0.0, 0.05)
    with tf3:
        search_query = st.text_input("Search Node ID", "")

    table_rows = []
    for n, pred in preds.items():
        conf = confs.get(n, 0.0)
        if filter_pred == "Sybils Only" and pred != 1:
            continue
        if filter_pred == "Normal Only" and pred != 0:
            continue
        if conf < min_conf:
            continue
        if search_query and search_query.lower() not in n.lower():
            continue

        deg = G.degree(n) if G else 0
        sus_nbrs = sum(1 for nbr in G.neighbors(n) if G.nodes[nbr].get("label") == 1) if G else 0
        true_lbl = str(gt[n]) if n in gt else "Unlabeled"

        table_rows.append({
            "Node ID": str(n),
            "Predicted Class": "Sybil" if pred == 1 else "Normal",
            "Confidence Score": round(float(conf), 4),
            "Degree": int(deg),
            "Suspicious Neighbors": int(sus_nbrs),
            "Ground Truth": true_lbl
        })

    table_df = pd.DataFrame(table_rows)
    st.markdown(f"**Showing {len(table_df):,} matching nodes:**")
    st.dataframe(table_df.head(100), hide_index=True, width="stretch")

    # Mathematical Formulation Documentation
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Mathematical Formulation & Algorithmic Foundation")
    st.latex(r"Y^{(t+1)} = \alpha P Y^{(t)} + (1 - \alpha) Y^{(0)}")
    st.markdown("""
    Where:
    - **$P = D^{-1} A$** represents the column/row-stochastic transition probability matrix.
    - **$\alpha \in (0, 1)$** is the probability of propagating trust from graph neighbors.
    - **$(1 - \alpha)$** is the probability of returning to the initial labeled priors $Y^{(0)}$.
    - **Hard Clamping:** $Y^{(t+1)}_{L} = Y^{(0)}_{L}$ preserves verified seed priors after every iteration.
    """)

    # Export Section
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Export Artifacts")

    export_data = []
    for n, pred in res["predictions"].items():
        conf = res["confidence_scores"].get(n, 0.0)
        true_lbl = str(gt[n]) if n in gt else "Unlabeled"
        deg = G.degree(n) if G else 0
        sus_nbrs = sum(1 for nbr in G.neighbors(n) if G.nodes[nbr].get("label") == 1) if G else 0

        export_data.append({
            "node_id": str(n),
            "ground_truth": true_lbl,
            "predicted_label": int(pred),
            "sybil_confidence": round(float(conf), 4),
            "degree": int(deg),
            "suspicious_neighbors_count": int(sus_nbrs)
        })

    df_export = pd.DataFrame(export_data)
    csv_buf = io.StringIO()
    df_export.to_csv(csv_buf, index=False)

    json_sum = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": st.session_state.current_dataset_name,
        "total_nodes": len(df_export),
        "predicted_sybils": int((df_export["predicted_label"] == 1).sum()),
        "predicted_normal": int((df_export["predicted_label"] == 0).sum()),
        "device_info": device_info,
    }

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="Download Predictions CSV",
            data=csv_buf.getvalue(),
            file_name="graphshield_predictions.csv",
            mime="text/csv",
            width="stretch"
        )
    with dl2:
        st.download_button(
            label="Download Run Summary JSON",
            data=json.dumps(json_sum, indent=2),
            file_name="graphshield_run_summary.json",
            mime="application/json",
            width="stretch"
        )

    # Pipeline CTAs
    st.markdown("---")
    c_back5, c_restart = st.columns([1, 2])
    with c_back5:
        if st.button("<- Back to Evaluation", width="stretch"):
            st.session_state.pipeline_step = "4. Evaluation"
            st.rerun()
    with c_restart:
        if st.button("Restart Pipeline with New Dataset", type="primary", width="stretch"):
            st.session_state.pipeline_step = "1. Select Dataset"
            st.rerun()
