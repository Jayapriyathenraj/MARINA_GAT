"""
Module 3m: Land-Cell Handling Strategy Comparison

Tests whether zero-filling land cells (the default) shapes results, by
comparing it against dropping land cells entirely and against zero-fill
plus an explicit binary validity-mask feature. GAT and GCN are retrained
under all three strategies, spatial block protocol, 5 seeds x 5 folds each.

LEAKAGE FIX (same as evaluation_splits.py's build_train_subgraph): training
previously forwarded the FULL graph (train + held-out nodes together) and
only masked the loss to train_idx, letting held-out node FEATURES leak into
train representations via GATConv/GCNConv's neighbor aggregation. Fix: the
training loop now forwards only the node-induced subgraph over train_idx.
The full graph is still used at EVAL time only (inference under a fixed
model).

CONFIG-PARITY FIX: this module previously used its own from-scratch
protocol (uniform-random node folds, lr=0.01, epochs=100,
seed=seed*100+fold) that silently diverged from evaluation_splits.py's
spatial_block protocol used for the main Table 4 results -- the same
class of mismatch found and fixed in topology_ablation.py and
hyperparameter_sensitivity.py. That meant the "zero-fill (default)" row
here (nominally the same graph/model/split as Table 4's GAT/GCN rows) was
not actually comparable to Table 4. This version imports the exact
split/train routines from evaluation_splits.py instead of reimplementing
them, so all three strategy rows here are now produced by the identical
protocol used for Table 4, and "zero-fill (default)" should now agree
with Table 4's GAT/GCN numbers (up to run-to-run noise).
"""
import numpy as np
import pandas as pd
import torch
from graph_construction import build_graph
from evaluation_splits import assign_blocks, train_and_eval_gat, train_and_eval_gcn, N_FOLDS, SEEDS

STRATEGIES = ["zero_fill", "remove", "valid_mask"]
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
                "mse": mse, "mae": mae,
            })
            print(f"seed={seed} fold={fold} {model_cls_name}: MSE={mse:.4f} MAE={mae:.4f}")
    return rows


def run_land_handling_comparison():
    rows = []
    for strategy in STRATEGIES:
        graph = build_graph("ocean_physics.nc", land_handling=strategy)
        for name, trainer in TRAINERS.items():
            print(f"\n=== [{strategy}] {name} (spatial-block protocol) ===")
            model_rows = _spatial_block_mses(graph, name, trainer)
            for r in model_rows:
                r["strategy"] = strategy
            rows.extend(model_rows)

    df = pd.DataFrame(rows)
    df.to_csv("figures/land_handling_results.csv", index=False)

    summary = df.groupby(["strategy", "model"]).agg(
        mse_mean=("mse", "mean"), mae_mean=("mae", "mean"), n_runs=("mse", "count")
    ).reset_index()
    print("\n=== Land-Handling Summary ===")
    print(summary.to_string(index=False))
    print("\nSaved figures/land_handling_results.csv")
    return df


if __name__ == "__main__":
    run_land_handling_comparison()
