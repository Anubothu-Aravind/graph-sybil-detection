"""Interactive graph visualization and diagnostic charting using PyVis and Plotly."""

from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network


# Design Palette
COLOR_MAP = {
    "known_normal": "#10B981",       # Emerald Green
    "known_sybil": "#EF4444",        # Crimson Red
    "predicted_normal": "#38BDF8",   # Sky Blue
    "predicted_sybil": "#F59E0B",    # Vibrant Amber / Orange
    "unlabeled": "#64748B",          # Slate Gray
}


def get_node_visual_category(
    node: str,
    known_labels: Dict[str, int],
    predictions: Optional[Dict[str, int]] = None
) -> Tuple[str, str, str]:
    """Determine visualization category, color, and friendly label for a node."""
    known_lbl = known_labels.get(node, -1)
    pred_lbl = predictions.get(node, -1) if predictions else -1

    if known_lbl == 1:
        return "known_sybil", COLOR_MAP["known_sybil"], "Known Sybil (Training Seed)"
    elif known_lbl == 0:
        return "known_normal", COLOR_MAP["known_normal"], "Known Normal (Training Seed)"
    elif pred_lbl == 1:
        return "predicted_sybil", COLOR_MAP["predicted_sybil"], "Predicted Sybil"
    elif pred_lbl == 0:
        return "predicted_normal", COLOR_MAP["predicted_normal"], "Predicted Normal"
    else:
        return "unlabeled", COLOR_MAP["unlabeled"], "Unlabeled Account"


def build_interactive_graph_html(
    G: nx.Graph,
    known_labels: Dict[str, int],
    predictions: Optional[Dict[str, int]] = None,
    confidence_scores: Optional[Dict[str, float]] = None,
    filter_mode: str = "all",
    focus_node: Optional[str] = None,
    max_nodes: int = 150,
    height: str = "600px",
    width: str = "100%"
) -> Tuple[str, Dict[str, Any]]:
    """Build interactive HTML visualization using PyVis.

    Supports responsive filtering for large networks:
      - 'all': Renders graph up to max_nodes (with intelligent sampling if exceeded).
      - 'sybils_and_neighbors': Renders only known/predicted Sybils and their 1-hop neighbors.
      - 'ego': Renders 1-hop or 2-hop neighborhood of a specified focus_node.
      - 'suspicious_only': Renders only predicted and known Sybils.

    Returns:
        html_code: Complete raw HTML string ready for Streamlit components.html.
        stats: Summary of visualized nodes and edges.
    """
    total_nodes = G.number_of_nodes()

    # Determine subset of nodes to display
    display_nodes: Set[str] = set()

    if filter_mode == "ego" and focus_node and focus_node in G:
        display_nodes = set(nx.single_source_shortest_path_length(G, focus_node, cutoff=2).keys())
    elif filter_mode == "sybils_and_neighbors":
        sybil_set = {
            n for n in G.nodes()
            if known_labels.get(n) == 1 or (predictions and predictions.get(n) == 1)
        }
        display_nodes = set(sybil_set)
        for s in sybil_set:
            display_nodes.update(G.neighbors(s))
    elif filter_mode == "suspicious_only":
        display_nodes = {
            n for n in G.nodes()
            if known_labels.get(n) == 1 or (predictions and predictions.get(n) == 1)
        }
    else:
        # Default: all nodes, with sampling if too large
        if total_nodes <= max_nodes:
            display_nodes = set(G.nodes())
        else:
            # Prioritize Sybils and highest degree nodes to keep view informative
            priority_nodes = [
                n for n in G.nodes()
                if known_labels.get(n) == 1 or (predictions and predictions.get(n) == 1)
            ]
            other_nodes = sorted(
                [n for n in G.nodes() if n not in priority_nodes],
                key=lambda x: G.degree(x),
                reverse=True
            )
            selected = priority_nodes + other_nodes[:max(0, max_nodes - len(priority_nodes))]
            display_nodes = set(selected[:max_nodes])

    # Ensure display_nodes is not empty
    if not display_nodes:
        display_nodes = set(list(G.nodes())[:min(max_nodes, total_nodes)])

    subgraph = G.subgraph(display_nodes)

    # Initialize PyVis network
    net = Network(height=height, width=width, bgcolor="#0f172a", font_color="#f8fafc")
    net.toggle_physics(True)

    # Configure physics for smooth layout and stability
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": { "size": 13, "face": "Inter, system-ui, sans-serif", "color": "#f8fafc" }
      },
      "edges": {
        "color": { "color": "rgba(148, 163, 184, 0.35)", "highlight": "#38bdf8", "hover": "#94a3b8" },
        "smooth": { "type": "continuous", "roundness": 0.2 },
        "width": 1.2
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.015,
          "springLength": 80,
          "springConstant": 0.08,
          "damping": 0.4
        },
        "maxVelocity": 40,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "enabled": true, "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": false,
        "zoomView": true
      }
    }
    """)

    # Add nodes with custom styling and tooltips
    for node in subgraph.nodes():
        cat, color, label_desc = get_node_visual_category(node, known_labels, predictions)
        conf = confidence_scores.get(node, 0.0) if confidence_scores else 0.0
        deg = G.degree(node)
        suspicious_nbrs = sum(1 for nbr in G.neighbors(node) if known_labels.get(nbr) == 1)

        # Tooltip HTML
        title_html = (
            f"<div style='font-family: system-ui; font-size: 13px; line-height: 1.5; color: #1e293b; padding: 4px;'>"
            f"<b>Node:</b> {node}<br/>"
            f"<b>Status:</b> <span style='font-weight:600;'>{label_desc}</span><br/>"
            f"<b>Sybil Confidence:</b> {conf * 100:.1f}%<br/>"
            f"<b>Degree:</b> {deg}<br/>"
            f"<b>Suspicious Neighbors:</b> {suspicious_nbrs}"
            f"</div>"
        )

        size = 14 + min(18, int(deg * 1.5))
        if cat in ["known_sybil", "predicted_sybil"]:
            size += 4

        # Border color
        border_color = "#ffffff" if node == focus_node else ("#fca5a5" if cat.endswith("sybil") else "#94a3b8")

        net.add_node(
            node,
            label=str(node),
            title=title_html,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": color, "border": "#38bdf8"},
                "hover": {"background": color, "border": "#ffffff"},
            },
            size=size,
            shape="dot"
        )

    # Add edges
    for u, v in subgraph.edges():
        net.add_edge(u, v)

    html_code = net.generate_html()

    stats = {
        "nodes_displayed": len(display_nodes),
        "edges_displayed": subgraph.number_of_edges(),
        "total_graph_nodes": total_nodes,
        "sampled": len(display_nodes) < total_nodes,
    }

    return html_code, stats


def plot_degree_distribution(G: nx.Graph) -> go.Figure:
    """Plot interactive degree distribution histogram and statistics."""
    degrees = [d for _, d in G.degree()]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=degrees,
        nbinsx=max(10, min(35, len(set(degrees)))),
        marker=dict(color="#38bdf8", line=dict(color="#0284c7", width=1)),
        opacity=0.85,
        name="Node Degree"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        title=dict(text="<b>Node Degree Distribution</b>", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title="Degree (Number of Connections)", gridcolor="#334155", color="#cbd5e1"),
        yaxis=dict(title="Count of Nodes", gridcolor="#334155", color="#cbd5e1"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig


def plot_confusion_matrix(cm_dict: Dict[str, int]) -> go.Figure:
    """Plot interactive confusion matrix heatmap."""
    tn, fp, fn, tp = cm_dict.get("tn", 0), cm_dict.get("fp", 0), cm_dict.get("fn", 0), cm_dict.get("tp", 0)
    z = [[tn, fp], [fn, tp]]
    x_labels = ["Pred Normal (0)", "Pred Sybil (1)"]
    y_labels = ["True Normal (0)", "True Sybil (1)"]

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=f"<b>{val}</b>",
                showarrow=False,
                font=dict(color="#ffffff", size=18)
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#0f172a"], [0.5, "#0284c7"], [1, "#0284c7"]],
        showscale=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        title=dict(text="<b>Held-Out Test Confusion Matrix</b>", font=dict(size=16, color="#f8fafc")),
        annotations=annotations,
        margin=dict(l=50, r=50, t=50, b=50),
        height=320,
        xaxis=dict(color="#cbd5e1"),
        yaxis=dict(color="#cbd5e1"),
    )
    return fig


def plot_roc_curve(fpr: List[float], tpr: List[float], roc_auc: float) -> go.Figure:
    """Plot ROC Curve with AUC score annotation."""
    fig = go.Figure()

    # ROC Line
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        line=dict(color="#f59e0b", width=3),
        name=f"GraphShield (AUC = {roc_auc:.3f})"
    ))

    # Diagonal Random Baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color="#64748b", dash="dash"),
        name="Random Classifier (AUC = 0.500)"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        title=dict(text=f"<b>Receiver Operating Characteristic (ROC)</b>", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title="False Positive Rate (FPR)", gridcolor="#334155", color="#cbd5e1", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (TPR)", gridcolor="#334155", color="#cbd5e1", range=[-0.02, 1.02]),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        legend=dict(x=0.45, y=0.1, bgcolor="rgba(15, 23, 42, 0.6)"),
    )
    return fig


def plot_confidence_distribution(confidences: Dict[str, float]) -> go.Figure:
    """Plot distribution of continuous Sybil confidence scores."""
    scores = list(confidences.values())
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker=dict(color="#ec4899", line=dict(color="#be185d", width=1)),
        opacity=0.85,
        name="Sybil Confidence"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        title=dict(text="<b>Sybil Confidence Score Distribution</b>", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title="Predicted Sybil Probability", gridcolor="#334155", color="#cbd5e1", range=[-0.02, 1.02]),
        yaxis=dict(title="Account Count", gridcolor="#334155", color="#cbd5e1"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig
