"""
Module 3f: Generates publication-ready figures + a metrics table from the
GAT vs LinReg (vs GCN, if present) comparison. Run this AFTER compare_models.py
so it can reuse the same trained models and test split.

LEAKAGE FIX (vs. original): target mean/std were previously computed from
raw_target.mean()/.std() over ALL nodes (train + test) before the split,
then applied to both sides -- test-set values leaked into the scale used
for the training loss. Fixed the same way evaluation_splits.py fixes it:
stats are computed from train_idx only and applied to both train and test.
Test-side numbers now reflect genuinely unseen-scale generalization, not
just unseen values on a scale the model was indirectly told about.

Produces (saved to ./figures/):
  - metrics_bar_chart.png   (grouped bar chart: MSE, MAE per model)
  - loss_curves.png         (train/test loss vs epoch, per model)
  - pred_vs_true_scatter.png (held-out predictions vs ground truth)
  - results_table.png       (clean table image for direct paper insertion)
  - results_table.csv       (same table, raw values for LaTeX/Word import)
"""
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from graph_construction import build_graph
from gat_model import MarinaGAT
from linreg_baseline import LinearBaseline, train_linreg

try:
    from gcn_model import MarinaGCN, train_gcn
    HAS_GCN = True
except ImportError:
    HAS_GCN = False

os.makedirs("figures", exist_ok=True)
torch.manual_seed(42)

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

results = {}  # name -> dict(mse, mae, train_losses, test_losses, pred)


def eval_model(model, name, is_linreg=False):
    model.eval()
    with torch.no_grad():
        pred = model(graph.x) if is_linreg else model(graph.x, graph.edge_index)
        if isinstance(pred, tuple):
            pred = pred[0]
    test_pred = pred[test_idx].squeeze()
    test_true = target[test_idx].squeeze()
    mse = F.mse_loss(test_pred, test_true).item()
    mae = F.l1_loss(test_pred, test_true).item()
    results[name] = {"mse": mse, "mae": mae,
                      "pred": test_pred.numpy(), "true": test_true.numpy()}
    return mse, mae


# --- Train GAT, logging per-epoch losses ---
print("=== Training GAT ===")
gat_model = MarinaGAT(in_channels=graph.x.shape[1] - 1)
optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)
gat_train_losses, gat_test_losses, gat_epochs = [], [], []
for epoch in range(100):
    gat_model.train()
    optimizer.zero_grad()
    pred = gat_model(graph.x, graph.edge_index)
    if isinstance(pred, tuple):
        pred = pred[0]
    loss = F.mse_loss(pred[train_idx], target[train_idx])
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        gat_model.eval()
        with torch.no_grad():
            tp = gat_model(graph.x, graph.edge_index)
            if isinstance(tp, tuple):
                tp = tp[0]
            tloss = F.mse_loss(tp[test_idx], target[test_idx])
        gat_epochs.append(epoch)
        gat_train_losses.append(loss.item())
        gat_test_losses.append(tloss.item())
eval_model(gat_model, "GAT")
results["GAT"]["train_curve"] = (gat_epochs, gat_train_losses)
results["GAT"]["test_curve"] = (gat_epochs, gat_test_losses)

# --- GCN (optional, only if the file is still present in this Colab session) ---
if HAS_GCN:
    print("\n=== Training GCN ===")
    gcn_model = train_gcn(graph, train_idx, test_idx, target)
    eval_model(gcn_model, "GCN")

# --- Linear Regression ---
print("\n=== Training Linear Regression ===")
linreg_model = train_linreg(graph, train_idx, test_idx, target)
eval_model(linreg_model, "LinReg", is_linreg=True)

# ==== everything below is unchanged from the original file ====

import pandas as pd
import numpy as np

model_names = list(results.keys())

# --- metrics_bar_chart.png ---
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
mses = [results[m]["mse"] for m in model_names]
maes = [results[m]["mae"] for m in model_names]
ax[0].bar(model_names, mses, color=["#2b6cb0", "#48bb78", "#a0aec0"][:len(model_names)])
ax[0].set_title("Test MSE"); ax[0].set_ylabel("MSE (normalized)")
ax[1].bar(model_names, maes, color=["#2b6cb0", "#48bb78", "#a0aec0"][:len(model_names)])
ax[1].set_title("Test MAE"); ax[1].set_ylabel("MAE (normalized)")
fig.tight_layout()
fig.savefig("figures/metrics_bar_chart.png", dpi=200)
plt.close(fig)

# --- loss_curves.png ---
if "train_curve" in results.get("GAT", {}):
    fig, ax = plt.subplots(figsize=(6, 4))
    ep, tr = results["GAT"]["train_curve"]
    _, te = results["GAT"]["test_curve"]
    ax.plot(ep, tr, label="GAT train")
    ax.plot(ep, te, label="GAT test")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss"); ax.legend()
    fig.tight_layout()
    fig.savefig("figures/loss_curves.png", dpi=200)
    plt.close(fig)

# --- pred_vs_true_scatter.png ---
fig, axes = plt.subplots(1, len(model_names), figsize=(4.5 * len(model_names), 4.2), sharey=True)
if len(model_names) == 1:
    axes = [axes]
for ax, name in zip(axes, model_names):
    ax.scatter(results[name]["true"], results[name]["pred"], alpha=0.6, s=15)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_title(name)
    ax.set_xlabel("True (normalized)")
axes[0].set_ylabel("Predicted (normalized)")
fig.tight_layout()
fig.savefig("figures/pred_vs_true_scatter.png", dpi=200)
plt.close(fig)

# --- results_table.csv / .png ---
table_df = pd.DataFrame({
    "Model": model_names,
    "Test MSE": [results[m]["mse"] for m in model_names],
    "Test MAE": [results[m]["mae"] for m in model_names],
})
table_df.to_csv("figures/results_table.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 1 + 0.4 * len(model_names)))
ax.axis("off")
tbl = ax.table(cellText=table_df.round(4).values, colLabels=table_df.columns,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
fig.tight_layout()
fig.savefig("figures/results_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved figures/metrics_bar_chart.png, loss_curves.png, "
      "pred_vs_true_scatter.png, results_table.png, results_table.csv")
print(table_df)
