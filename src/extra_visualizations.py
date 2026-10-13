"""
Module 3i: Extra Visualizations for the Paper
Generates additional publication-ready figures beyond results_plots.py and
evaluation_splits.py's summary bar chart:

  1. mse_heatmap.png        - split x model heatmap of mean MSE
  2. attention_distribution.png - histogram of GAT attention weights
  3. spatial_prediction_map.png - lat/lon scatter, node color = predicted
                                   biodiversity score (GAT), from the
                                   spatial-only graph
  4. spatial_error_map.png  - lat/lon scatter, node color = |pred - true|
  5. attention_map.png      - lat/lon scatter, node color = mean incoming
                              attention weight per node (which regions the
                              model treats as most influential)

Run AFTER evaluation_splits.py (needs figures/all_splits_summary.csv) and
after gat_model.py / explainability.py have been run at least once (needs
marina_gat.pt, and reuses graph_construction.py directly).

Usage:
    python extra_visualizations.py
Produces (saved to ./figures/):
    mse_heatmap.png
    attention_distribution.png
    spatial_prediction_map.png
    spatial_error_map.png
    attention_map.png
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from graph_construction import build_graph, FEATURE_COLUMNS
from gat_model import MarinaGAT

os.makedirs("figures", exist_ok=True)
sns.set_style("whitegrid")

MODEL_ORDER = ["GAT", "GCN", "LinReg"]
SPLIT_ORDER = ["temporal", "spatial_block"]


# ---------- 1 & 2: heatmaps from all_splits_summary.csv ----------

def plot_metric_heatmap(summary_csv="figures/all_splits_summary.csv"):
    df = pd.read_csv(summary_csv)

    mse_pivot = df.pivot(index="split", columns="model", values="mse_mean").reindex(
        index=SPLIT_ORDER, columns=MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.heatmap(mse_pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax,
                cbar_kws={"label": "Mean MSE (lower = better)"}, linewidths=0.5)
    ax.set_title("Mean Test MSE by Split and Model")
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig("figures/mse_heatmap.png", dpi=200)
    plt.close(fig)

    print("Saved figures/mse_heatmap.png")


# ---------- helpers: load trained GAT + spatial graph ----------

def load_gat_and_graph(weights_path="marina_gat.pt", physics_nc="ocean_physics.nc"):
    graph = build_graph(physics_nc)
    model = MarinaGAT(in_channels=graph.x.shape[1] - 1)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        preds, (attn1, attn2) = model(graph.x, graph.edge_index, return_attention=True)
    return graph, model, preds, attn1


# ---------- 3: attention weight distribution ----------

def plot_attention_distribution(attn1):
    edge_index_out, alpha = attn1
    alpha_mean = alpha.mean(dim=1).numpy()
    src, dst = edge_index_out[0].numpy(), edge_index_out[1].numpy()
    non_self = src != dst
    weights = alpha_mean[non_self]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(weights, bins=40, color="#2b6cb0", edgecolor="white")
    ax.axvline(np.median(weights), color="#e53e3e", linestyle="--",
               label=f"median = {np.median(weights):.3f}")
    ax.set_xlabel("Attention weight (layer 1, head-averaged, excl. self-loops)")
    ax.set_ylabel("Edge count")
    ax.set_title("Distribution of GAT Attention Weights over Current-Informed Edges")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/attention_distribution.png", dpi=200)
    plt.close(fig)
    print("Saved figures/attention_distribution.png")


# ---------- 4 & 5: spatial prediction / error maps ----------

def plot_spatial_prediction_and_error(graph, preds):
    coords = graph.coords.reset_index(drop=True)
    pred = preds.squeeze().numpy()
    raw_target = graph.x[:, -1:].squeeze().numpy()
    target_norm = (raw_target - raw_target.mean()) / (raw_target.std() + 1e-6)
    valid = graph.x[:, 0].numpy() != 0.0  # drop zeroed land/missing cells

    lat, lon = coords["lat"].values, coords["lon"].values

    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(lon[valid], lat[valid], c=pred[valid], cmap="viridis", s=18)
    fig.colorbar(sc, ax=ax, label="Predicted biodiversity score (normalized)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("GAT Predicted Biodiversity Score by Location")
    fig.tight_layout()
    fig.savefig("figures/spatial_prediction_map.png", dpi=200)
    plt.close(fig)

    err = np.abs(pred - target_norm)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(lon[valid], lat[valid], c=err[valid], cmap="Reds", s=18)
    fig.colorbar(sc, ax=ax, label="|Predicted - True| (normalized)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("GAT Absolute Prediction Error by Location")
    fig.tight_layout()
    fig.savefig("figures/spatial_error_map.png", dpi=200)
    plt.close(fig)

    print("Saved figures/spatial_prediction_map.png, figures/spatial_error_map.png")


# ---------- 6: spatial attention map (mean incoming attention per node) ----------

def plot_attention_map(graph, attn1):
    edge_index_out, alpha = attn1
    alpha_mean = alpha.mean(dim=1).numpy()
    dst = edge_index_out[1].numpy()
    src = edge_index_out[0].numpy()
    non_self = src != dst

    n = graph.num_nodes
    incoming = np.zeros(n)
    counts = np.zeros(n)
    for d, w in zip(dst[non_self], alpha_mean[non_self]):
        incoming[d] += w
        counts[d] += 1
    counts[counts == 0] = 1
    incoming_mean = incoming / counts

    coords = graph.coords.reset_index(drop=True)
    valid = graph.x[:, 0].numpy() != 0.0
    lat, lon = coords["lat"].values, coords["lon"].values

    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(lon[valid], lat[valid], c=incoming_mean[valid], cmap="magma", s=18)
    fig.colorbar(sc, ax=ax, label="Mean incoming attention weight")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Spatial Distribution of GAT Attention (Explainability Map)")
    fig.tight_layout()
    fig.savefig("figures/attention_map.png", dpi=200)
    plt.close(fig)
    print("Saved figures/attention_map.png")


def run_all():
    print("=== 1. Heatmap from evaluation_splits.py summary ===")
    if os.path.exists("figures/all_splits_summary.csv"):
        plot_metric_heatmap()
    else:
        print("figures/all_splits_summary.csv not found -- run evaluation_splits.py first. Skipping heatmap.")

    print("\n=== Loading trained GAT + spatial graph for attention/map figures ===")
    try:
        graph, model, preds, attn1 = load_gat_and_graph()
    except FileNotFoundError as e:
        print(f"Skipping attention/map figures -- missing file: {e}")
        return

    print("\n=== 2. Attention weight distribution ===")
    plot_attention_distribution(attn1)

    print("\n=== 3-4. Spatial prediction and error maps ===")
    plot_spatial_prediction_and_error(graph, preds)

    print("\n=== 5. Spatial attention map ===")
    plot_attention_map(graph, attn1)

    print("\nAll extra figures saved to ./figures/")


if __name__ == "__main__":
    run_all()
