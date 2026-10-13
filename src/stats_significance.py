"""
Module 3k: Statistical Significance Testing

Paired Wilcoxon signed-rank tests, bootstrap 95% confidence intervals, and
Cohen's d for the model/topology comparisons most central to this paper's
claims, computed directly from the per-fold paired MSE values in
figures/all_splits_results.csv and figures/topology_ablation_results.csv
(NOT from the aggregated summary files), so fold-level variance is
controlled for rather than compared across independent runs.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

np.random.seed(0)


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05):
    diffs = np.asarray(diffs)
    n = len(diffs)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = diffs[np.random.randint(0, n, n)]
        boots[b] = sample.mean()
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return lo, hi


def cohens_d(diffs):
    diffs = np.asarray(diffs)
    return float(diffs.mean() / diffs.std(ddof=1))


def paired_report(a, b, name, rows):
    a = np.asarray(a)
    b = np.asarray(b)
    diffs = a - b
    stat, p = wilcoxon(a, b)
    d = cohens_d(diffs)
    lo, hi = bootstrap_ci(diffs)
    print(f"{name}: n={len(a)} mean_diff={diffs.mean():.4f} 95%CI=[{lo:.4f},{hi:.4f}] "
          f"wilcoxon_p={p:.4f} cohend={d:.3f}")
    rows.append({
        "comparison": name, "n": len(a), "mean_diff_mse": diffs.mean(),
        "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p,
        "significant_05": p < 0.05, "cohens_d": d,
    })


def run_stats_significance():
    rows = []

    splits = pd.read_csv("figures/all_splits_results.csv")

    def get(df, key_cols, split_val, model_val):
        sub = df[(df[key_cols[0]] == split_val) & (df["model"] == model_val)]
        sub = sub.sort_values(["seed", "fold"])
        return sub["mse"].values

    gat_sp = get(splits, ["split"], "spatial_block", "GAT")
    gcn_sp = get(splits, ["split"], "spatial_block", "GCN")
    lin_sp = get(splits, ["split"], "spatial_block", "LinReg")

    print("=== Spatial block (n=25) ===")
    paired_report(gat_sp, gcn_sp, "GAT vs. GCN (spatial block)", rows)
    paired_report(gat_sp, lin_sp, "GAT vs. LinReg (spatial block)", rows)

    try:
        topo = pd.read_csv("figures/topology_ablation_results.csv")

        def get2(topology_val, model_val):
            sub = topo[(topo["topology"] == topology_val) & (topo["model"] == model_val)]
            sub = sub.sort_values(["seed", "fold"])
            return sub["mse"].values

        gat_c = get2("Current-informed", "GAT")
        gcn_c = get2("Current-informed", "GCN")
        gat_k = get2("Geographic-kNN", "GAT")
        gcn_k = get2("Geographic-kNN", "GCN")

        print("\n=== Topology ablation (n=25 each) ===")
        paired_report(gat_c, gcn_c, "GAT vs. GCN (current-informed)", rows)
        paired_report(gat_c, gat_k, "GAT: current-informed vs. geographic-kNN", rows)
        paired_report(gcn_c, gcn_k, "GCN: current-informed vs. geographic-kNN", rows)
    except FileNotFoundError:
        print("\ntopology_ablation_results.csv not found -- run topology_ablation.py first "
              "to include those comparisons.")

    out = pd.DataFrame(rows)
    out.to_csv("figures/stats_pairwise.csv", index=False)
    print("\nSaved figures/stats_pairwise.csv")
    return out


if __name__ == "__main__":
    run_stats_significance()
