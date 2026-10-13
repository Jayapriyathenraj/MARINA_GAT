"""
Module 3g: Spatial block cross-validation for GAT vs GCN vs LinReg.

WHY THIS EXISTS (replaces compare_models.py's single random node split):
Grid-cell nodes are spatially autocorrelated -- a node at (13.0N, 80.2E) has
nearly identical temp/salinity/current values to its neighbor at (13.05N,
80.2E). A *random* train/test split over nodes puts near-duplicate cells on
both sides of the split, so a model can "cheat" by memorizing local
conditions instead of learning anything generalizable. Reported test
MSE from a random split will look better than the model actually is.

This module instead:
  1. Bins nodes into spatial blocks (a coarse lat/lon grid over the study
     area, independent of the graph's own node grid).
  2. Assigns whole blocks -- never individual nodes -- to train or test,
     so no test node is a near-neighbor of a train node.
  3. Repeats across multiple block-fold assignments AND multiple random
     seeds (which reseed model init + block-to-fold assignment), and
     reports mean +/- std, not a single point estimate.

LEAKAGE FIXES (both were present in the original version of this file):
  a) Message passing across the train/test boundary. GATConv/GCNConv
     aggregate each node's representation from its graph neighbors. Simply
     computing loss on pred[train_idx] while forwarding the FULL graph
     (train_idx and test_idx together) lets held-out node FEATURES leak
     into train node representations during training, even though held-out
     LABELS never enter the loss. Fix: during training, forward only the
     node-induced subgraph over train_idx (build_train_subgraph below), so
     test nodes and their edges are structurally absent from the training
     computation graph. At eval time we forward the full graph -- that's
     fine and realistic (a deployed model gets to see surrounding regions'
     features at inference time; it just never trained on this fold's
     held-out labels or feature-derived gradients).
  b) Target normalization computed from train+test together. mean()/std()
     over the whole graph's target column let test-set target values shape
     the scale used for the training loss. Fix: normalize using train-only
     statistics (train_target_stats), applied to both train and test.

Usage:
    python spatial_cv.py
Produces:
    figures/spatial_cv_results.csv   (per-run rows: model, seed, fold, mse, mae)
    figures/spatial_cv_summary.csv   (mean +/- std per model)
    figures/spatial_cv_summary.png   (bar chart with error bars)
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

from graph_construction import build_graph
from gat_model import MarinaGAT
from gcn_model import MarinaGCN
from linreg_baseline import LinearBaseline

N_LAT_BLOCKS = 4      # coarse spatial binning grid for CV blocks
N_LON_BLOCKS = 4
N_FOLDS = 5            # blocks are partitioned into this many held-out folds
SEEDS = [0, 1, 2, 3, 4]
EPOCHS_GAT = 250
EPOCHS_OTHER = 100


def assign_blocks(coords_df, n_lat_blocks=N_LAT_BLOCKS, n_lon_blocks=N_LON_BLOCKS):
    """Bin each node into a coarse (lat_bin, lon_bin) spatial block id."""
    lat_bins = pd.cut(coords_df["lat"], bins=n_lat_blocks, labels=False)
    lon_bins = pd.cut(coords_df["lon"], bins=n_lon_blocks, labels=False)
    block_id = lat_bins.astype(int) * n_lon_blocks + lon_bins.astype(int)
    return block_id.values


def spatial_kfold_splits(block_ids, n_folds, seed):
    """
    Yield (train_idx, test_idx) node-index arrays, n_folds times, where
    every node in a given spatial block always lands entirely in either
    the train or the test side of a given fold.
    """
    rng = np.random.RandomState(seed)
    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)

    block_folds = np.array_split(unique_blocks, n_folds)

    for k in range(n_folds):
        test_blocks = set(block_folds[k].tolist())
        test_mask = np.array([b in test_blocks for b in block_ids])
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield torch.tensor(train_idx, dtype=torch.long), torch.tensor(test_idx, dtype=torch.long)


def compute_metrics(pred, true):
    pred = pred.squeeze()
    true = true.squeeze()
    mse = F.mse_loss(pred, true).item()
    mae = F.l1_loss(pred, true).item()
    return mse, mae


def train_target_stats(raw_target, idx_train):
    """Mean/std computed from TRAIN nodes only -- never from test nodes."""
    train_vals = raw_target[idx_train]
    return train_vals.mean(), train_vals.std() + 1e-6


def build_train_subgraph(graph, idx_train):
    """
    Node-induced subgraph over idx_train only: test nodes and any edge
    touching a test node are structurally absent, so a training forward
    pass cannot let GAT/GCN message-pass test-node features into train
    representations. relabel_nodes=True renumbers surviving nodes
    0..len(idx_train)-1 in the order of idx_train, which matches the row
    order of graph.x[idx_train] used below.
    """
    edge_index_train, _ = subgraph(
        idx_train, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
    )
    x_train = graph.x[idx_train]
    return x_train, edge_index_train


def train_and_eval_gat(graph, train_idx, test_idx, raw_target, seed):
    torch.manual_seed(seed)
    mean, std = train_target_stats(raw_target, train_idx)
    target = (raw_target - mean) / std

    x_train, edge_index_train = build_train_subgraph(graph, train_idx)
    target_train = target[train_idx]

    model = MarinaGAT(in_channels=graph.x.shape[1] - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    for _ in range(EPOCHS_GAT):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train, edge_index_train)
        loss = F.mse_loss(pred, target_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Full graph at eval time is fine: this is inference, not training --
        # no gradient step ever sees test-node information.
        pred_full = model(graph.x, graph.edge_index)
    return compute_metrics(pred_full[test_idx], target[test_idx])


def train_and_eval_gcn(graph, train_idx, test_idx, raw_target, seed):
    torch.manual_seed(seed)
    mean, std = train_target_stats(raw_target, train_idx)
    target = (raw_target - mean) / std

    x_train, edge_index_train = build_train_subgraph(graph, train_idx)
    target_train = target[train_idx]

    model = MarinaGCN(in_channels=graph.x.shape[1] - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(EPOCHS_OTHER):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train, edge_index_train)
        loss = F.mse_loss(pred, target_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_full = model(graph.x, graph.edge_index)
    return compute_metrics(pred_full[test_idx], target[test_idx])


def train_and_eval_linreg(graph, train_idx, test_idx, raw_target, seed):
    # No message passing here (no graph convolution), so there's no
    # feature-leakage channel to fix -- but keep train-only normalization
    # for consistency with the other two models.
    torch.manual_seed(seed)
    mean, std = train_target_stats(raw_target, train_idx)
    target = (raw_target - mean) / std

    model = LinearBaseline(in_channels=graph.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(EPOCHS_OTHER):
        model.train()
        optimizer.zero_grad()
        pred = model(graph.x[train_idx])
        loss = F.mse_loss(pred, target[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_full = model(graph.x)
    return compute_metrics(pred_full[test_idx], target[test_idx])


def run_spatial_cv():
    os.makedirs("figures", exist_ok=True)
    graph = build_graph("ocean_physics.nc")
    raw_target = graph.x[:, -1:].clone()  # NOTE: kept unnormalized here -- each
    # fold now normalizes using its own train_idx inside the trainers above.

    block_ids = assign_blocks(graph.coords)
    n_blocks = len(np.unique(block_ids))
    print(f"Spatial CV: {graph.num_nodes} nodes grouped into {n_blocks} spatial blocks, "
          f"{N_FOLDS} folds x {len(SEEDS)} seeds")
    if n_blocks < N_FOLDS:
        print(f"WARNING: only {n_blocks} distinct spatial blocks but N_FOLDS={N_FOLDS}. "
              "Some folds will be skipped -- results below are still valid but come "
              "from fewer than N_FOLDS x len(SEEDS) runs. Consider a smaller grid "
              "(fewer nodes) needs coarser N_LAT_BLOCKS/N_LON_BLOCKS, or a larger "
              "study area / finer node grid needs more blocks.")

    rows = []
    trainers = {"GAT": train_and_eval_gat, "GCN": train_and_eval_gcn, "LinReg": train_and_eval_linreg}

    for seed in SEEDS:
        for fold, (train_idx, test_idx) in enumerate(spatial_kfold_splits(block_ids, N_FOLDS, seed)):
            for name, trainer in trainers.items():
                mse, mae = trainer(graph, train_idx, test_idx, raw_target, seed)
                rows.append({"model": name, "seed": seed, "fold": fold,
                             "n_train": len(train_idx), "n_test": len(test_idx),
                             "mse": mse, "mae": mae})
                print(f"[{name}] seed={seed} fold={fold} "
                      f"n_train={len(train_idx)} n_test={len(test_idx)} "
                      f"MSE={mse:.4f} MAE={mae:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("figures/spatial_cv_results.csv", index=False)

    summary = df.groupby("model").agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"), mse_median=("mse", "median"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"), mae_median=("mae", "median"),
        n_runs=("mse", "size"),
    ).reset_index()
    summary.to_csv("figures/spatial_cv_summary.csv", index=False)
    print("\n=== Spatial CV summary across folds x seeds ===")
    for _, r in summary.iterrows():
        print(f"{r['model']:>7} | MSE: {r['mse_mean']:.4f} +/- {r['mse_std']:.4f} "
              f"(median {r['mse_median']:.4f}) "
              f"| MAE: {r['mae_mean']:.4f} +/- {r['mae_std']:.4f} "
              f"(median {r['mae_median']:.4f}) "
              f"| n_runs: {int(r['n_runs'])}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = {"GAT": "#2b6cb0", "GCN": "#48bb78", "LinReg": "#a0aec0"}
    for ax, metric, title in zip(
        axes, ["mse", "mae"],
        ["Test MSE (lower better)", "Test MAE (lower better)"],
    ):
        means = [summary.loc[summary["model"] == m, f"{metric}_mean"].values[0] for m in trainers]
        stds = [summary.loc[summary["model"] == m, f"{metric}_std"].values[0] for m in trainers]
        ax.bar(list(trainers.keys()), means, yerr=stds, capsize=4,
               color=[colors[m] for m in trainers])
        ax.set_title(title, fontsize=10)

    fig.suptitle(f"Spatial block CV, {N_FOLDS} folds x {len(SEEDS)} seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/spatial_cv_summary.png", dpi=200)
    plt.close(fig)
    print("\nSaved figures/spatial_cv_results.csv, spatial_cv_summary.csv, spatial_cv_summary.png")
    return df, summary


if __name__ == "__main__":
    run_spatial_cv()
