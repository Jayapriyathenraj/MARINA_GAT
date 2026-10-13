"""
Module 3l: Hyperparameter Sensitivity Analysis

Checks whether the reported results are an artifact of a fortunate default
setting for the three graph-construction hyperparameters (angular
tolerance, max_edges_per_node, drift threshold), all previously described
as empirically chosen but untuned. GAT is retrained across a small grid of
alternative values for each hyperparameter in turn, holding the other two
at their default, under the spatial block protocol (5 seeds x 5 folds).

LEAKAGE FIX (same as evaluation_splits.py's build_train_subgraph): training
previously forwarded the FULL graph (train + held-out nodes together) and
only masked the loss to train_idx, letting held-out node FEATURES leak into
train representations via GATConv's neighbor aggregation. Fix: the training
loop now forwards only the node-induced subgraph over train_idx. The full
graph is still used at EVAL time only (inference under a fixed model).

CONFIG-PARITY FIX: this module previously used its own from-scratch
protocol (uniform-random node folds, lr=0.01, epochs=100, seed=seed*100+fold)
that silently diverged from evaluation_splits.py's spatial_block protocol
used for the main GAT result in Table 4 (block-CV folds from assign_blocks,
lr=0.005, epochs=min(EPOCHS_GAT, 120), seed=seed). That mismatch is what
made the "default" row of this table (MSE ~10.072) incomparable to Table 4's
GAT MSE (3.544), even though both were nominally the same default
hyperparameters. This version imports the exact split/train routine from
evaluation_splits.py instead of reimplementing it, so the default row here
is now produced by the identical protocol used for Table 4, and the two
numbers should now agree (up to run-to-run noise).
"""
import numpy as np
import pandas as pd
import torch
from graph_construction import build_graph
from evaluation_splits import assign_blocks, train_and_eval_gat, N_FOLDS, SEEDS

DEFAULT_ANGLE = 45.0
DEFAULT_MAX_EDGES = 5
DEFAULT_DRIFT = 0.02

SWEEP = {
    "angle_tolerance_deg": [15.0, 30.0, 45.0, 60.0, 90.0],
    "max_edges_per_node": [2, 3, 5, 8, 12],
    "drift_threshold": [0.0, 0.01, 0.02, 0.05, 0.1],
}


def _spatial_block_gat_mses(graph):
    """
    Same spatial-block-CV protocol as evaluation_splits.run_spatial_block,
    restricted to the GAT model, so results here are directly comparable
    to Table 4's GAT row.
    """
    raw_target = graph.x[:, -1:].clone()
    block_ids = assign_blocks(graph.coords)
    mses = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        blocks = np.unique(block_ids)
        rng.shuffle(blocks)
        folds = np.array_split(blocks, N_FOLDS)
        for test_blocks in folds:
            test_mask = np.isin(block_ids, test_blocks)
            idx_test = torch.tensor(np.where(test_mask)[0], dtype=torch.long)
            idx_train = torch.tensor(np.where(~test_mask)[0], dtype=torch.long)
            if len(idx_test) == 0 or len(idx_train) == 0:
                continue
            mse, _ = train_and_eval_gat(
                graph, idx_train, graph, idx_test, raw_target, raw_target, seed
            )
            mses.append(mse)
    return np.array(mses)


def run_hyperparameter_sensitivity():
    rows = []
    for hyperparam, values in SWEEP.items():
        for value in values:
            kwargs = dict(angle_tolerance_deg=DEFAULT_ANGLE,
                           max_edges_per_node=DEFAULT_MAX_EDGES,
                           drift_threshold=DEFAULT_DRIFT)
            kwargs[hyperparam] = value
            is_default = (value == {"angle_tolerance_deg": DEFAULT_ANGLE,
                                     "max_edges_per_node": DEFAULT_MAX_EDGES,
                                     "drift_threshold": DEFAULT_DRIFT}[hyperparam])

            graph = build_graph("ocean_physics.nc", **kwargs)
            mses = _spatial_block_gat_mses(graph)
            rows.append({
                "hyperparam": hyperparam, "value": value, "is_default": is_default,
                "mse_mean": mses.mean(), "mse_std": mses.std(),
                "n_edges": graph.num_edges,
            })
            print(f"{hyperparam}={value} (default={is_default}): "
                  f"MSE={mses.mean():.4f}+-{mses.std():.4f} n_edges={graph.num_edges}")

    df = pd.DataFrame(rows)
    df.to_csv("figures/hyperparam_sensitivity_results.csv", index=False)
    print("\nSaved figures/hyperparam_sensitivity_results.csv")
    return df


if __name__ == "__main__":
    run_hyperparameter_sensitivity()
