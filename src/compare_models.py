"""
Module 3e: GAT vs Linear Regression comparison on the identical
train/test split -- an ablation check for whether the current-informed
graph (with attention) beats a no-graph baseline.

LEAKAGE FIX (vs. original): target mean/std were previously computed from
raw_target.mean()/.std() over ALL nodes (train + test) before the split.
Fixed to use train-only stats, same convention as evaluation_splits.py.
"""
import torch
import torch.nn.functional as F
from graph_construction import build_graph
from gat_model import MarinaGAT
from linreg_baseline import LinearBaseline, train_linreg

torch.manual_seed(42)  # fix split so every model sees the exact same nodes

graph = build_graph("ocean_physics.nc")

raw_target = graph.x[:, -1:].clone()

n = graph.num_nodes
perm = torch.randperm(n)
train_idx = perm[: int(0.8 * n)]
test_idx = perm[int(0.8 * n):]

# --- LEAKAGE FIX: normalize using TRAIN stats only, apply to both sides ---
train_mean = raw_target[train_idx].mean()
train_std = raw_target[train_idx].std() + 1e-6
target = (raw_target - train_mean) / train_std

print("=== Training GAT ===")
# in_channels - 1: species_richness excluded inside MarinaGAT.forward()
gat_model = MarinaGAT(in_channels=graph.x.shape[1] - 1)
optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(100):
    gat_model.train()
    optimizer.zero_grad()
    pred = gat_model(graph.x, graph.edge_index)
    loss = F.mse_loss(pred[train_idx], target[train_idx])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        gat_model.eval()
        with torch.no_grad():
            test_pred = gat_model(graph.x, graph.edge_index)
            if isinstance(test_pred, tuple):
                test_pred = test_pred[0]
            test_loss = F.mse_loss(test_pred[test_idx], target[test_idx])
        print(f"[GAT] Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

print("\n=== Training Linear Regression (no-graph baseline) ===")
linreg_model = train_linreg(graph, train_idx, test_idx, target)


def final_metrics(model, name):
    model.eval()
    with torch.no_grad():
        if isinstance(model, LinearBaseline):
            pred = model(graph.x)
        else:
            pred = model(graph.x, graph.edge_index)
        if isinstance(pred, tuple):
            pred = pred[0]
    test_pred = pred[test_idx].squeeze()
    test_true = target[test_idx].squeeze()
    mse = F.mse_loss(test_pred, test_true).item()
    mae = F.l1_loss(test_pred, test_true).item()
    print(f"{name:>7} | Test MSE: {mse:.4f} | Test MAE: {mae:.4f}")
    return mse, mae


print("\n=== Final comparison (held-out test nodes) ===")
final_metrics(gat_model, "GAT")
final_metrics(linreg_model, "LinReg")
