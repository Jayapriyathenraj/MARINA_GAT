"""
Module 3h: Evaluation splits, from weakest to strongest test of generalization.

  1. Spatial block split  -- k-fold over spatial blocks (what spatial_cv.py
     already did). Blocks rotate through train/test; every block is tested
     on exactly once per seed. Tests: "does this generalize to nearby but
     unseen locations, on average?"

  2. Temporal split        -- train on node features from an EARLIER time
     window, test on the SAME nodes' features from a LATER window (same
     species_richness target throughout, since richness is computed once
     from the full OBIS record set and is not re-collected per window).
     Tests: "does this generalize forward in
     time, at known locations?" Requires copernicus_ingest.py's
     fetch_early_late_windows() to have been run.

LEAKAGE FIXES (both present in the original version of this file):
  a) Message passing across the train/test boundary. Split 1 forwarded the
     FULL graph (train and held-out nodes together) during training and
     only masked the LOSS to train_idx. Since GATConv/GCNConv aggregate
     each node from its graph neighbors, that let held-out node FEATURES
     leak into train representations during training. Fix: train_and_eval_*
     now forward only the node-induced subgraph over idx_train (see
     build_train_subgraph) during training, and use the full graph only at
     eval time (fine -- that's inference, not a gradient step). Split 2
     (temporal) already used two physically separate graph objects for
     train vs. test, so it never had this problem.
  b) Target normalization computed from train+test together (split 1
     called normalized_target(graph) once over the whole graph before
     splitting). Fix: normalization stats are now computed from idx_train
     only and applied to both sides of the split.

Both report MSE/MAE mean+-std -- same conventions as spatial_cv.py so
results are comparable across split types.

Usage:
    python evaluation_splits.py
Produces:
    figures/all_splits_results.csv     (every individual run)
    figures/all_splits_summary.csv     (per split-type x model)
    figures/all_splits_summary.png     (mean MSE by split type, GAT vs GCN vs LinReg)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.utils import subgraph

from graph_construction import build_graph, align_graphs_by_coords
from gat_model import MarinaGAT
from gcn_model import MarinaGCN
from linreg_baseline import LinearBaseline

N_LAT_BLOCKS = 4
N_LON_BLOCKS = 4
N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
EPOCHS_GAT = 250
EPOCHS_OTHER = 100

TRAINERS = {}  # filled in below, model name -> train_and_eval fn


# ---------- shared helpers (same conventions as spatial_cv.py) ----------

def assign_blocks(coords_df, n_lat_blocks=N_LAT_BLOCKS, n_lon_blocks=N_LON_BLOCKS):
    lat_bins = pd.cut(coords_df["lat"], bins=n_lat_blocks, labels=False)
    lon_bins = pd.cut(coords_df["lon"], bins=n_lon_blocks, labels=False)
    return (lat_bins.astype(int) * n_lon_blocks + lon_bins.astype(int)).values


def compute_metrics(pred, true):
    pred, true = pred.squeeze(), true.squeeze()
    mse = F.mse_loss(pred, true).item()
    mae = F.l1_loss(pred, true).item()
    return mse, mae


def normalized_target(graph):
    """
    Used ONLY for split 2 (temporal), where train and test are already two
    physically separate graphs (graph_early / graph_late) -- normalizing
    each with its own stats doesn't feed test information back into any
    training gradient, it only rescales how that side's metrics are
    reported. Split 1 uses train_target_stats() below instead, since
    there train and test share one graph/index space.
    """
    raw = graph.x[:, -1:].clone()
    return (raw - raw.mean()) / (raw.std() + 1e-6)


def train_target_stats(raw_target, idx_train):
    """Mean/std from TRAIN indices only -- never from held-out indices."""
    train_vals = raw_target[idx_train]
    return train_vals.mean(), train_vals.std() + 1e-6


def build_train_subgraph(graph, idx_train):
    """
    Node-induced subgraph over idx_train: held-out nodes and any edge
    touching one are structurally absent, so training can't message-pass
    held-out features into train representations. relabel_nodes=True
    renumbers surviving nodes 0..len(idx_train)-1 in idx_train's order,
    matching graph.x[idx_train]'s row order.
    """
    edge_index_train, _ = subgraph(
        idx_train, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
    )
    x_train = graph.x[idx_train]
    return x_train, edge_index_train


def train_and_eval_gat(graph_train, idx_train, graph_test, idx_test, raw_target_train, raw_target_test, seed):
    torch.manual_seed(seed)

    # Train side always normalizes from train indices only, whether train
    # and test share one graph (split 1) or are different graphs (split 2,
    # temporal).
    mean, std = train_target_stats(raw_target_train, idx_train)
    target_train_full = (raw_target_train - mean) / std
    target_test_full = (raw_target_test - raw_target_test.mean()) / (raw_target_test.std() + 1e-6) \
        if raw_target_train is not raw_target_test else target_train_full

    x_train, edge_index_train = build_train_subgraph(graph_train, idx_train)
    target_train = target_train_full[idx_train]

    model = MarinaGAT(in_channels=graph_train.x.shape[1] - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    n_epochs = min(EPOCHS_GAT, 120)
    for _ in range(n_epochs):
        model.train(); optimizer.zero_grad()
        pred = model(x_train, edge_index_train)
        loss = F.mse_loss(pred, target_train)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_full = model(graph_test.x, graph_test.edge_index)
    return compute_metrics(pred_full[idx_test], target_test_full[idx_test])


def train_and_eval_gcn(graph_train, idx_train, graph_test, idx_test, raw_target_train, raw_target_test, seed):
    torch.manual_seed(seed)
    mean, std = train_target_stats(raw_target_train, idx_train)
    target_train_full = (raw_target_train - mean) / std
    target_test_full = (raw_target_test - raw_target_test.mean()) / (raw_target_test.std() + 1e-6) \
        if raw_target_train is not raw_target_test else target_train_full

    x_train, edge_index_train = build_train_subgraph(graph_train, idx_train)
    target_train = target_train_full[idx_train]

    model = MarinaGCN(in_channels=graph_train.x.shape[1] - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(EPOCHS_OTHER):
        model.train(); optimizer.zero_grad()
        pred = model(x_train, edge_index_train)
        loss = F.mse_loss(pred, target_train)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_full = model(graph_test.x, graph_test.edge_index)
    return compute_metrics(pred_full[idx_test], target_test_full[idx_test])


def train_and_eval_linreg(graph_train, idx_train, graph_test, idx_test, raw_target_train, raw_target_test, seed):
    # No graph convolution -> no message-passing leak channel, but keep
    # train-only normalization for consistency/comparability.
    torch.manual_seed(seed)
    mean, std = train_target_stats(raw_target_train, idx_train)
    target_train_full = (raw_target_train - mean) / std
    target_test_full = (raw_target_test - raw_target_test.mean()) / (raw_target_test.std() + 1e-6) \
        if raw_target_train is not raw_target_test else target_train_full

    model = LinearBaseline(in_channels=graph_train.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(EPOCHS_OTHER):
        model.train(); optimizer.zero_grad()
        pred = model(graph_train.x[idx_train])
        loss = F.mse_loss(pred, target_train_full[idx_train])
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_full = model(graph_test.x)
    return compute_metrics(pred_full[idx_test], target_test_full[idx_test])


TRAINERS = {"GAT": train_and_eval_gat, "GCN": train_and_eval_gcn, "LinReg": train_and_eval_linreg}


# ---------- split 1: spatial block k-fold ----------

def run_spatial_block(graph, raw_target, rows):
    block_ids = assign_blocks(graph.coords)
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        blocks = np.unique(block_ids); rng.shuffle(blocks)
        folds = np.array_split(blocks, N_FOLDS)
        for fold, test_blocks in enumerate(folds):
            test_mask = np.isin(block_ids, test_blocks)
            idx_test = torch.tensor(np.where(test_mask)[0], dtype=torch.long)
            idx_train = torch.tensor(np.where(~test_mask)[0], dtype=torch.long)
            if len(idx_test) == 0 or len(idx_train) == 0:
                continue
            for name, trainer in TRAINERS.items():
                mse, mae = trainer(graph, idx_train, graph, idx_test, raw_target, raw_target, seed)
                rows.append({"split": "spatial_block", "model": name, "seed": seed, "fold": fold,
                             "n_train": len(idx_train), "n_test": len(idx_test),
                             "mse": mse, "mae": mae})


# ---------- split 2: temporal (train early window, test late window) ----------

def run_temporal(graph_early, graph_late, rows):
    # Two physically separate graph objects -- no message-passing leak
    # possible here. Each side normalizes with its own stats (see
    # normalized_target docstring above for why that's not a training leak).
    target_early = normalized_target(graph_early)
    target_late = normalized_target(graph_late)
    idx_all = torch.arange(graph_early.num_nodes)
    for seed in SEEDS:
        for name, trainer in TRAINERS.items():
            mse, mae = trainer(graph_early, idx_all, graph_late, idx_all,
                                graph_early.x[:, -1:].clone(), graph_late.x[:, -1:].clone(), seed)
            rows.append({"split": "temporal", "model": name, "seed": seed, "fold": 0,
                         "n_train": len(idx_all), "n_test": len(idx_all),
                         "mse": mse, "mae": mae})


def summarize(df):
    summary = df.groupby(["split", "model"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        n_runs=("mse", "size"),
    ).reset_index()
    return summary


def plot_summary(summary, splits_present):
    fig, axes = plt.subplots(1, len(splits_present), figsize=(4.2 * len(splits_present), 4.2), sharey=True)
    if len(splits_present) == 1:
        axes = [axes]
    colors = {"GAT": "#2b6cb0", "GCN": "#48bb78", "LinReg": "#a0aec0"}
    for ax, split in zip(axes, splits_present):
        sub = summary[summary["split"] == split]
        names = [m for m in ["GAT", "GCN", "LinReg"] if m in sub["model"].values]
        means = [sub.loc[sub["model"] == m, "mse_mean"].values[0] for m in names]
        bars = ax.bar(names, means, color=[colors[m] for m in names])
        ax.set_title(split.replace("_", " "), fontsize=10)
    fig.suptitle("Test MSE mean by evaluation strictness", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/all_splits_summary.png", dpi=200)
    plt.close(fig)


def run_all(temporal_available=True):
    os.makedirs("figures", exist_ok=True)
    rows = []

    print("=== Building spatial-only graph (single window) ===")
    graph = build_graph("ocean_physics.nc")
    raw_target = graph.x[:, -1:].clone()

    print("\n=== 1. Spatial block split ===")
    run_spatial_block(graph, raw_target, rows)

    if temporal_available:
        try:
            print("\n=== Building early/late graphs for temporal split ===")
            graph_early_raw = build_graph("ocean_physics_early.nc")
            graph_late_raw = build_graph("ocean_physics_late.nc")
            graph_early, graph_late = align_graphs_by_coords(graph_early_raw, graph_late_raw)
            print(f"Aligned to {graph_early.num_nodes} common nodes across early/late windows")

            print("\n=== 2. Temporal split ===")
            run_temporal(graph_early, graph_late, rows)
        except FileNotFoundError:
            print("\nocean_physics_early.nc / ocean_physics_late.nc not found -- skipping "
                  "temporal split. Run copernicus_ingest.py's fetch_early_late_windows() "
                  "first to enable it.")
        except ValueError as e:
            print(f"\nSkipping temporal split: {e}")

    df = pd.DataFrame(rows)
    df.to_csv("figures/all_splits_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv("figures/all_splits_summary.csv", index=False)

    print("\n=== Summary ===")
    for _, r in summary.iterrows():
        print(f"{r['split']:>16} | {r['model']:>7} | MSE {r['mse_mean']:.4f}+-{r['mse_std']:.4f} "
              f"| MAE {r['mae_mean']:.4f}+-{r['mae_std']:.4f} "
              f"| n_runs {int(r['n_runs'])}")

    splits_present = [s for s in ["spatial_block", "temporal"]
                       if s in summary["split"].values]
    plot_summary(summary, splits_present)
    print(f"\nSaved figures/all_splits_results.csv, all_splits_summary.csv, all_splits_summary.png")
    return df, summary


if __name__ == "__main__":
    run_all()
