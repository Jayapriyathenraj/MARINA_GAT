"""
Module 4: Explainability
Extracts GAT attention weights and links them to real oceanographic drivers.
This is the paper's key XAI contribution: attention isn't just a number,
it's tied to WHY a region matters (strong current, upwelling-like temp/salinity
gradient, etc.)

Updated to use the time-aggregated feature columns from graph_construction.py
(temp_mean, salinity_mean, u_mean, v_mean, ...) instead of the old single-
snapshot columns (temp, salinity, u, v).
"""
import torch
import pandas as pd
import numpy as np
from graph_construction import build_graph, FEATURE_COLUMNS
from gat_model import MarinaGAT


def extract_attention(model, graph):
    """
    Run inference and pull out attention weights from the first GAT layer
    (most interpretable - shows raw region-to-region importance).
    """
    model.eval()
    with torch.no_grad():
        preds, (attn1, attn2) = model(graph.x, graph.edge_index, return_attention=True)

    edge_index_out, alpha = attn1  # alpha shape: [num_edges, num_heads]
    alpha_mean = alpha.mean(dim=1)  # average across attention heads

    src = edge_index_out[0].numpy()
    dst = edge_index_out[1].numpy()

    return pd.DataFrame({
        "source_node": src,
        "target_node": dst,
        "attention_weight": alpha_mean.numpy(),
    }), preds


def explain_top_edges(attn_df, graph, node_df, top_k=10):
    """
    For the highest-attention edges, explain WHY in oceanographic terms:
    - strong mean current magnitude -> active drift/transport pathway
    - temperature/salinity gradient -> possible upwelling or mixing zone
    - high temp_std/u_std/v_std -> a volatile node (unstable conditions),
      worth flagging separately since it changes how much to trust the
      time-mean edge direction there
    """
    # Remove self-loops (GATConv auto-adds these internally; they always get
    # inflated attention ~1.0 and aren't meaningful region-to-region connections)
    attn_df = attn_df[attn_df["source_node"] != attn_df["target_node"]].reset_index(drop=True)

    # Remove edges touching zeroed-out land/missing-data cells (temp_mean=0 is
    # not a real ocean reading here - it's our NaN placeholder from
    # graph_construction.py)
    valid_nodes = node_df[node_df["temp_mean"] != 0.0].index
    attn_df = attn_df[
        attn_df["source_node"].isin(valid_nodes) & attn_df["target_node"].isin(valid_nodes)
    ].reset_index(drop=True)

    top_edges = attn_df.sort_values("attention_weight", ascending=False).head(top_k).copy()

    explanations = []
    for _, row in top_edges.iterrows():
        i, j = int(row.source_node), int(row.target_node)

        temp_i, sal_i, u_i, v_i, temp_std_i = node_df.iloc[i][
            ["temp_mean", "salinity_mean", "u_mean", "v_mean", "temp_std"]
        ]
        temp_j, sal_j = node_df.iloc[j][["temp_mean", "salinity_mean"]]

        current_mag = np.sqrt(u_i**2 + v_i**2)
        temp_gradient = abs(temp_i - temp_j)
        sal_gradient = abs(sal_i - sal_j)

        reasons = []
        if current_mag > 0.1:
            reasons.append(f"strong mean current ({current_mag:.2f} m/s) driving connectivity")
        if temp_gradient > 0.5:
            reasons.append(f"temperature gradient ({temp_gradient:.2f}\u00b0C) suggesting upwelling/mixing")
        if sal_gradient > 0.3:
            reasons.append(f"salinity gradient ({sal_gradient:.2f} PSU)")
        if temp_std_i > 0.5:
            reasons.append(f"source node is thermally volatile over the window (std={temp_std_i:.2f}\u00b0C) "
                            f"-- treat this edge's direction as less stable")
        if not reasons:
            reasons.append("weak but consistent drift pathway")

        lat_i, lon_i = graph.coords.iloc[i][["lat", "lon"]]
        lat_j, lon_j = graph.coords.iloc[j][["lat", "lon"]]

        explanations.append({
            "region": f"({lat_i:.3f},{lon_i:.3f}) -> ({lat_j:.3f},{lon_j:.3f})",
            "attention_weight": round(row.attention_weight, 4),
            "explanation": "; ".join(reasons),
        })

    return pd.DataFrame(explanations)


if __name__ == "__main__":
    graph = build_graph("ocean_physics.nc")
    model = MarinaGAT(in_channels=graph.x.shape[1] - 1)  # matches gat_model.py training convention
    model.load_state_dict(torch.load("marina_gat.pt"))

    node_df = pd.DataFrame(graph.x.numpy(), columns=FEATURE_COLUMNS)

    attn_df, preds = extract_attention(model, graph)
    print(f"Extracted attention for {len(attn_df)} edges\n")

    explained = explain_top_edges(attn_df, graph, node_df, top_k=10)
    print("Top 10 most influential region-to-region connections:\n")
    print(explained.to_string(index=False))

    explained.to_csv("explainability_report.csv", index=False)
    print("\nSaved to explainability_report.csv")
