"""
Module 2: Graph Construction with Current-Informed Edges
Builds a PyTorch Geometric graph where:
  - Nodes = grid cells across the marina zone, each with TIME-AGGREGATED
            ocean features computed across the full downloaded window
            (mean, variability, and trend per variable) rather than a
            single snapshot -- this is what gives the model something to
            learn beyond "the currents happened to look like X on one day".
  - Edges = connect regions ALONG the direction ocean currents actually flow,
            using the time-mean current (not simple geographic proximity)
            -- this is your novelty. Edges are symmetrized after selection
            and every node is guaranteed max_edges_per_node edges via a
            per-node geographic-nearest-neighbor fallback/top-up, so no
            node is left near-isolated and connectivity is directly
            comparable to a geographic-kNN baseline graph (see
            build_geographic_knn_edges below).

TARGET VARIABLE: species_richness (unique species observed near each node),
NOT species_count (raw OBIS occurrence-record count). An earlier version of
this pipeline used species_count as the prediction target, which conflates
true species diversity with historical sampling effort: a single heavily
observed species contributes to that count identically to fifty distinct
species each observed once. species_richness fixes this by counting DISTINCT
species per node instead. The raw occurrence count is still computed and
kept as a diagnostic-only column, occurrence_record_count, for the
sampling-effort analysis in the paper -- it is never fed into any model.
"""
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch_geometric.data import Data

# Node feature columns produced by build_node_features, in order.
# (species_richness is always last -- every model in this project excludes
# the last column as the prediction target; keep it last if you add features.)
FEATURE_COLUMNS = [
    "temp_mean", "temp_std", "temp_trend",
    "salinity_mean", "salinity_std",
    "u_mean", "u_std",
    "v_mean", "v_std",
    "current_speed_mean", "current_speed_std",
    "species_richness",
]


def _linear_trend(values):
    """Slope of a simple linear fit over the time axis; 0.0 if <2 valid points."""
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return 0.0
    t = np.arange(len(values))[valid]
    y = values[valid]
    slope, _ = np.polyfit(t, y, 1)
    return float(slope)


def build_node_features(physics_nc="ocean_physics.nc", obis_csv=None):
    """
    Create one node per (lat, lon) grid cell from the Copernicus time window.
    Each node feature = time-aggregated [temp, salinity, u, v, current speed]
    statistics (mean/std/trend) + species_richness, instead of a single
    snapshot's raw values. This requires copernicus_ingest.py to have pulled
    more than one timestep (see DEFAULT_WINDOW_DAYS there); if the file only
    has a single timestep, std/trend will come out as 0.0 for every node,
    which is a sign the ingest step needs a wider date range.
    """
    ds = xr.open_dataset(physics_nc)
    ds = ds.isel(depth=0) if "depth" in ds.dims else ds

    has_time = "time" in ds.sizes and ds.sizes["time"] > 1
    if not has_time:
        print("WARNING: only one timestep in ocean_physics.nc -- temporal "
              "features (std, trend) will be degenerate (all zero). Re-run "
              "copernicus_ingest.py with a wider date range to fix this.")

    lats = ds.latitude.values
    lons = ds.longitude.values

    nodes = []
    coords = []
    land_mask = []  # True = land/no-data cell (zero-filled), False = ocean cell
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if has_time:
                temp_series = ds.thetao.values[:, i, j]
                sal_series = ds.so.values[:, i, j]
                u_series = ds.uo.values[:, i, j]
                v_series = ds.vo.values[:, i, j]
            else:
                temp_series = np.array([ds.thetao.values[i, j]])
                sal_series = np.array([ds.so.values[i, j]])
                u_series = np.array([ds.uo.values[i, j]])
                v_series = np.array([ds.vo.values[i, j]])

            # land / missing-data guard: if every timestep is NaN, this is a
            # land cell -- zero out rather than dropping, so the grid stays
            # rectangular for the edge-construction step
            if np.all(np.isnan(temp_series)):
                nodes.append([0.0] * (len(FEATURE_COLUMNS) - 1) + [0])
                coords.append((lat, lon))
                land_mask.append(True)
                continue

            speed_series = np.sqrt(u_series ** 2 + v_series ** 2)

            temp_mean = float(np.nanmean(temp_series))
            temp_std = float(np.nanstd(temp_series))
            temp_trend = _linear_trend(temp_series)
            sal_mean = float(np.nanmean(sal_series))
            sal_std = float(np.nanstd(sal_series))
            u_mean = float(np.nanmean(u_series))
            u_std = float(np.nanstd(u_series))
            v_mean = float(np.nanmean(v_series))
            v_std = float(np.nanstd(v_series))
            speed_mean = float(np.nanmean(speed_series))
            speed_std = float(np.nanstd(speed_series))

            species_richness = 0  # filled in below if obis_csv given
            nodes.append([temp_mean, temp_std, temp_trend, sal_mean, sal_std,
                          u_mean, u_std, v_mean, v_std, speed_mean, speed_std,
                          species_richness])
            coords.append((lat, lon))
            land_mask.append(False)

    node_df = pd.DataFrame(nodes, columns=FEATURE_COLUMNS)
    coords_df = pd.DataFrame(coords, columns=["lat", "lon"])
    coords_df["is_land"] = land_mask
    # Diagnostic-only column, never fed to any model: the raw OBIS
    # occurrence-record count at this node, kept alongside species_richness
    # so the sampling-effort confound can still be inspected and reported
    # without being the thing that is actually predicted.
    node_df["occurrence_record_count"] = 0

    # Optional: attach species richness (unique species count) per nearest
    # grid cell from OBIS data. species_richness -- NOT
    # occurrence_record_count -- is FEATURE_COLUMNS[-1] and is therefore the
    # actual prediction target used by every model in this pipeline.
    if obis_csv is not None:
        try:
            obis = pd.read_csv(obis_csv)
        except FileNotFoundError:
            print(f"Warning: {obis_csv} not found, species_richness will be 0 for all nodes")
            return node_df, coords_df

        species_col = None
        if "species" in obis.columns:
            species_col = "species"
        elif "scientificName" in obis.columns:
            species_col = "scientificName"
        if species_col is None:
            print("Warning: OBIS data has no 'species'/'scientificName' column -- "
                  "cannot compute richness, species_richness will be 0 for all nodes. "
                  "occurrence_record_count will still be populated for diagnostic use.")

        for idx, row in coords_df.iterrows():
            nearby = obis[
                (abs(obis["decimalLatitude"] - row["lat"]) < 0.3) &
                (abs(obis["decimalLongitude"] - row["lon"]) < 0.3)
            ]
            node_df.loc[idx, "occurrence_record_count"] = len(nearby)
            if species_col is not None:
                unique_species = (
                    nearby[species_col].dropna().astype(str).str.strip()
                )
                unique_species = unique_species[unique_species != ""]
                node_df.loc[idx, "species_richness"] = unique_species.nunique()

    return node_df, coords_df


def build_current_informed_edges(coords_df, node_df, drift_threshold=0.02,
                                  max_edges_per_node=5, angle_tolerance_deg=45.0):
    """
    THE NOVELTY STEP.
    Connect node i -> node j if the TIME-MEAN ocean current at node i points
    TOWARD node j (within angle_tolerance_deg), rather than just because
    they're geographically close.

    This means biodiversity-relevant signal (larval drift, pollutant
    transport, nutrient flow) propagates along actual water movement paths
    instead of straight-line distance.

    Every node -- even a stagnant one whose current magnitude falls below
    drift_threshold -- ends up with exactly max_edges_per_node edges: current-
    direction candidates are used where available, and topped up (or
    entirely replaced, for stagnant nodes) with nearest-geographic-neighbor
    edges otherwise. All edges are symmetrized (i -> j implies j -> i) after
    selection. This guarantees the current-informed graph has the same
    per-node degree as a geographic-kNN graph built with the same
    max_edges_per_node, so the two topologies are comparable on equal
    footing (see build_geographic_knn_edges and the topology ablation below).
    """
    n = len(coords_df)
    coords_arr = coords_df[["lat", "lon"]].values
    uv_arr = node_df[["u_mean", "v_mean"]].values

    # precompute full pairwise distance matrix once (n is a few thousand at most)
    diffs = coords_arr[:, None, :] - coords_arr[None, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=2)

    directed_pairs = set()

    for i in range(n):
        u, v = uv_arr[i]
        current_mag = np.sqrt(u**2 + v**2)

        chosen = np.array([], dtype=int)

        if current_mag >= drift_threshold:
            current_angle = np.arctan2(v, u)
            deltas = coords_arr - coords_arr[i]
            target_angles = np.arctan2(deltas[:, 0], deltas[:, 1])
            angle_diff = np.abs(np.arctan2(np.sin(target_angles - current_angle),
                                            np.cos(target_angles - current_angle)))
            tol = np.radians(angle_tolerance_deg)
            candidates = np.where(angle_diff < tol)[0]
            candidates = candidates[candidates != i]
            if len(candidates) > 0:
                dists = dist_matrix[i, candidates]
                chosen = candidates[np.argsort(dists)[:max_edges_per_node]]

        if len(chosen) < max_edges_per_node:
            # geographic-nearest-neighbor fallback/top-up -- guarantees every
            # node gets max_edges_per_node edges, same as a baseline graph
            dists_all = dist_matrix[i].copy()
            dists_all[i] = np.inf
            dists_all[chosen] = np.inf
            n_more = max_edges_per_node - len(chosen)
            extra = np.argsort(dists_all)[:n_more]
            chosen = np.concatenate([chosen, extra]).astype(int)

        for j in chosen:
            directed_pairs.add((i, int(j)))
            directed_pairs.add((int(j), i))  # symmetrize

    sym_pairs = sorted(directed_pairs)
    edge_index = torch.tensor(sym_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)

    return edge_index, edge_attr


def build_geographic_knn_edges(coords_df, max_edges_per_node=5):
    """
    Baseline topology for the ablation in Section 4.7: every node connects
    to its max_edges_per_node nearest geographic neighbors, with no
    reference to current direction at all. Edges are symmetrized the same
    way as build_current_informed_edges, so per-node degree matches exactly
    -- the only difference between the two graphs is WHICH neighbors are
    chosen, not HOW MANY.
    """
    n = len(coords_df)
    coords_arr = coords_df[["lat", "lon"]].values
    diffs = coords_arr[:, None, :] - coords_arr[None, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=2)

    directed_pairs = set()
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        chosen = np.argsort(dists)[:max_edges_per_node]
        for j in chosen:
            directed_pairs.add((i, int(j)))
            directed_pairs.add((int(j), i))

    sym_pairs = sorted(directed_pairs)
    edge_index = torch.tensor(sym_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
    return edge_index, edge_attr


def build_graph(physics_nc="ocean_physics.nc", obis_csv="obis_occurrences.csv",
                 topology="current_informed", land_handling="zero_fill",
                 drift_threshold=0.02, max_edges_per_node=5, angle_tolerance_deg=45.0):
    """
    topology: "current_informed" (default, novelty graph) or "geographic_knn"
              (baseline for the topology ablation).
    land_handling: "zero_fill" (default -- keep zero-filled land nodes in the
              grid), "remove" (drop land nodes before graph construction), or
              "valid_mask" (keep zero-filled land nodes but add a binary
              is_valid feature column so the model can learn to discount them).
    """
    node_df, coords_df = build_node_features(physics_nc, obis_csv)

    if land_handling == "remove":
        keep = ~coords_df["is_land"].values
        node_df = node_df.loc[keep].reset_index(drop=True)
        coords_df = coords_df.loc[keep].reset_index(drop=True)
    elif land_handling == "valid_mask":
        node_df["is_valid"] = (~coords_df["is_land"]).astype(float)
    elif land_handling != "zero_fill":
        raise ValueError(f"Unknown land_handling: {land_handling}")

    if topology == "current_informed":
        edge_index, edge_attr = build_current_informed_edges(
            coords_df, node_df, drift_threshold=drift_threshold,
            max_edges_per_node=max_edges_per_node,
            angle_tolerance_deg=angle_tolerance_deg)
    elif topology == "geographic_knn":
        edge_index, edge_attr = build_geographic_knn_edges(
            coords_df, max_edges_per_node=max_edges_per_node)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # feature columns actually fed to the model: FEATURE_COLUMNS, plus
    # is_valid if land_handling == "valid_mask". occurrence_record_count is
    # NEVER included here -- diagnostic only.
    model_columns = list(FEATURE_COLUMNS)
    if "is_valid" in node_df.columns:
        model_columns = model_columns[:-1] + ["is_valid"] + [model_columns[-1]]

    x = torch.tensor(node_df[model_columns].values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.coords = coords_df  # keep for later map plotting / explainability
    data.feature_names = model_columns
    if "occurrence_record_count" in node_df.columns:
        data.occurrence_record_count = torch.tensor(
            node_df["occurrence_record_count"].values, dtype=torch.float)

    print(f"Graph built: {data.num_nodes} nodes, {data.num_edges} {topology} edges, "
          f"{len(model_columns) - 1} input features/node (+ species_richness target); "
          f"land_handling={land_handling}")
    return data


def align_graphs_by_coords(graph_a, graph_b, tol=1e-6):
    """
    Needed for the temporal split: two graphs built from
    different .nc files (different time windows) can, in principle, come
    back with slightly different grids after independent coarsening. This
    intersects both graphs' node coordinates and returns REINDEXED
    (graph_a_aligned, graph_b_aligned) so node i in one corresponds to
    node i in the other. Edges are rebuilt for each aligned graph from its
    own (now node-count-matching) coordinates/features -- edges are NOT
    assumed to be shared, since current direction is time-window-specific.

    Raises if fewer than 10 nodes survive the intersection (not enough to
    do anything useful with).
    """
    a = graph_a.coords.round(6)
    b = graph_b.coords.round(6)
    merged = a.reset_index().merge(b.reset_index(), on=["lat", "lon"], suffixes=("_a", "_b"))
    if len(merged) < 10:
        raise ValueError(f"Only {len(merged)} nodes align by coordinate between the two "
                          f"graphs -- grids likely came back different shapes/extents. "
                          f"Re-check fetch_early_late_windows' bounding box/target_grid_size.")

    idx_a = merged["index_a"].values
    idx_b = merged["index_b"].values

    def _subset(graph, idx):
        x = graph.x[idx]
        coords = graph.coords.iloc[idx].reset_index(drop=True)
        node_df = pd.DataFrame(x.numpy(), columns=graph.feature_names)
        edge_index, edge_attr = build_current_informed_edges(coords, node_df)
        out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out.coords = coords
        out.feature_names = graph.feature_names
        return out

    return _subset(graph_a, idx_a), _subset(graph_b, idx_b)


if __name__ == "__main__":
    graph = build_graph("ocean_physics.nc")
    print(graph)
