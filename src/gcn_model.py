"""
Module 3b: Baseline GCN (no attention) for comparison against MarinaGAT.
Same graph, same task, same train loop shape — only the conv layer changes,
so any performance gap is attributable to attention, not to some other
confound (different features, different split, different optimizer).
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from graph_construction import build_graph


class MarinaGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        # drop species_richness (last column) - it's the prediction target,
        # not a feature (mirrors the exclusion in MarinaGAT / LinearBaseline)
        x = x[:, :-1]
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn2(x, edge_index))
        return self.out(x)


def train_gcn(graph, train_idx, test_idx, target, epochs=100, lr=0.01):
    # in_channels - 1: species_richness is excluded inside forward()
    model = MarinaGCN(in_channels=graph.x.shape[1] - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

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
            print(f"[GCN] Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

    return model
