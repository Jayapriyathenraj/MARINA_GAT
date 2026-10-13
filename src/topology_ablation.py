"""
Module 3j: Topology Ablation -- Current-Informed vs. Geographic-kNN

Isolates the specific contribution of current-informed edge DIRECTION from
the general contribution of having graph structure at all. GAT and GCN are
each retrained on two topologies that share identical per-node degree
(max_edges_per_node=5): the current-informed graph, and a geographic-kNN
graph with no reference to current direction. Everything else (features,
architecture, training procedure, spatial-block protocol) is held fixed.

LEAKAGE FIX (same as evaluation_splits.py's build_train_subgraph): training
previously forwarded the FULL graph (train + held-out nodes together) and
only masked the loss to train_idx. Since GATConv/GCNConv aggregate each
node from its graph neighbors, that let held-out node FEATURES leak into
train representations during training, even though the loss itself never
touched held-out labels. Fix: the training loop now forwards only the
node-induced subgraph over train_idx (held-out nodes and any edge touching
one are structurally absent). The full graph is still used at EVAL time
only -- that's inference under a fixed model, not a gradient step, so it
introduces no leakage.

CONFIG-PARITY FIX: this module previously used its own from-scratch
protocol (uniform-random node folds, lr=0.01, epochs=100,
seed=seed*100+fold) that silently diverged from evaluation_splits.py's
spatial_block protocol used for the main Table 4 results. That meant the
"Current-informed" row here (nominally the same graph/model/split as
Table 4's GAT row) was not actually comparable to it. This version imports
the exact split/train routines from evaluation_splits.py instead of
reimplementing them, so both topology rows here are now produced by the
identical protocol used for Table 4, and "Current-informed" should now
agree with Table 4's GAT/GCN numbers (up to run-to-run noise).
"""
import numpy as np
import pandas as pd
import torch
from graph_construction import build_graph
from evaluation_splits import assign_blocks, train_and_eval_gat, train_and_eval_gcn, N_FOLDS, SEEDS

TRAINERS = {"GAT": train_and_eval_gat, "GCN": train_and_eval_gcn}


def _spatial_block_mses(graph, model_cls_name, trainer):
    """
    Same spatial-block-CV protocol as evaluation_splits.run_spatial_block,
    for a single model, so results here are directly comparable to
    Table 4's per-model rows.
    """
    raw_target = graph.x[:, -1:].clone()
    block_ids = assign_blocks(graph.coords)
    rows = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        blocks = np.unique(block_ids)
        rng.shuffle(blocks)
        folds = np.array_split(blocks, N_FOLDS)
        for fold, test_blocks in enumerate(folds):
            test_mask = np.isin(block_ids, test_blocks)
            idx_test = torch.tensor(np.where(test_mask)[0], dtype=torch.long)
            idx_train = torch.tensor(np.where(~test_mask)[0], dtype=torch.long)
            if len(idx_test) == 0 or len(idx_train) == 0:
                continue
            mse, mae = trainer(graph, idx_train, graph, idx_test, raw_target, raw_target, seed)
            rows.append({
                "model": model_cls_name, "seed": seed, "fold": fold,
                "n_train": len(idx_train), "n_test": len(idx_test),
                "mse": mse, "mae": mae,
            })
            print(f"seed={seed} fold={fold} {model_cls_name}: MSE={mse:.4f} MAE={mae:.4f}")
    return rows


def run_topology_ablation():
    rows = []
    for topology in ["current_informed", "geographic_knn"]:
        graph = build_graph("ocean_physics.nc", topology=topology)
        topology_label = "Current-informed" if topology == "current_informed" else "Geographic-kNN"

        for name, trainer in TRAINERS.items():
            print(f"\n=== [{topology_label}] {name} (spatial-block protocol) ===")
            model_rows = _spatial_block_mses(graph, name, trainer)
            for r in model_rows:
                r["topology"] = topology_label
            rows.extend(model_rows)

    df = pd.DataFrame(rows)
    df.to_csv("figures/topology_ablation_results.csv", index=False)

    summary = df.groupby(["topology", "model"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        n_runs=("mse", "count"),
    ).reset_index()
    summary.to_csv("figures/topology_ablation_summary.csv", index=False)
    print("\n=== Topology Ablation Summary ===")
    print(summary.to_string(index=False))
    print("\nSaved figures/topology_ablation_results.csv, topology_ablation_summary.csv")


if __name__ == "__main__":
    run_topology_ablation()
