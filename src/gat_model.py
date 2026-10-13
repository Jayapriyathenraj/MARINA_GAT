"""
Module 3: Spatial Graph Attention Network (GAT)
Predicts an OBIS-derived species-richness signal (species_richness) per
region node, using
attention over current-informed edges. Attention weights double as
explainability output (Module 4 will visualize them).
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from graph_construction import build_graph


class MarinaGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=0.3)
        self.out = torch.nn.Linear(hidden_channels, 1)  # predict species_richness

    def forward(self, x, edge_index, return_attention=False):
        # drop species_richness (last column) - it's the prediction target, not
        # a feature. Leaving it in lets the model see the answer directly.
        x = x[:, :-1]
        x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        out = self.out(x)

        if return_attention:
            return out, (attn1, attn2)
        return out


def train_model(graph, epochs=100, lr=0.01):
    # in_channels - 1: species_richness is excluded inside forward()
    model = MarinaGAT(in_channels=graph.x.shape[1] - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    raw_target = graph.x[:, -1:].clone()

    # simple train/test split across nodes
    n = graph.num_nodes
    perm = torch.randperm(n)
    train_idx = perm[: int(0.8 * n)]
    test_idx = perm[int(0.8 * n):]

    # LEAKAGE FIX (vs. original): target used to be normalized with
    # raw_target.mean()/.std() over ALL nodes (train + test) before the
    # split existed. Now computed from train_idx only, same convention as
    # evaluation_splits.py, and applied to both train and test.
    train_mean = raw_target[train_idx].mean()
    train_std = raw_target[train_idx].std() + 1e-6
    target = (raw_target - train_mean) / train_std

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(graph.x, graph.edge_index)
        loss = F.mse_loss(pred[train_idx], target[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(graph.x, graph.edge_index)
                test_loss = F.mse_loss(test_pred[test_idx], target[test_idx])
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

    return model


if __name__ == "__main__":
    graph = build_graph("ocean_physics.nc")
    model = train_model(graph)

    # save for later use in dashboard / explainability module
    torch.save(model.state_dict(), "marina_gat.pt")
    print("\nModel saved to marina_gat.pt")

    # quick sanity check: run inference + get attention weights
    model.eval()
    with torch.no_grad():
        preds, attention = model(graph.x, graph.edge_index, return_attention=True)
    print(f"\nSample predictions (species richness) for first 5 nodes:")
    print(preds[:5].squeeze())
