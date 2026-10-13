"""
Module 3d: Linear Regression baseline — no graph, no message passing.
Predicts biodiversity score from each node's own raw features only
(temp, salinity, u, v). Ignores the current-informed edges completely.
This is the weakest sensible baseline: if GAT can't beat this, the
graph structure isn't earning its keep.
"""
import torch
import torch.nn.functional as F


class LinearBaseline(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # exclude species_richness (last column) - that's the target, not a feature
        self.linear = torch.nn.Linear(in_channels - 1, 1)

    def forward(self, x, edge_index=None):  # edge_index accepted but unused, kept for API symmetry
        feats = x[:, :-1]
        return self.linear(feats)


def train_linreg(graph, train_idx, test_idx, target, epochs=100, lr=0.01):
    model = LinearBaseline(in_channels=graph.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(graph.x)
        loss = F.mse_loss(pred[train_idx], target[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(graph.x)
                test_loss = F.mse_loss(test_pred[test_idx], target[test_idx])
            print(f"[LinReg] Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

    return model
