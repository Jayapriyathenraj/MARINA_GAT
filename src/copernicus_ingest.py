"""
Module 1c: Copernicus Marine Real-Time Data Ingestion
Pulls SST, salinity, and current vectors (u,v) for the study area.

Two modes:
  - fetch_ocean_physics(...): single window (used by the spatial-only splits).
  - fetch_early_late_windows(...): TWO non-overlapping windows -- "early"
    and "late" -- needed for the temporal split. You
    train node features from the early window and test on the late window,
    which is the only honest way to ask "does this generalize forward in
    time" (a random or spatial split within one window can't answer that
    at all -- it never trains on the past and predicts the future).

SETUP (one-time, do this first in Colab):
  !pip install copernicusmarine
  Register free account: https://data.marine.copernicus.eu/register
  Then run: copernicusmarine login   (enter username/password when prompted)
"""
import copernicusmarine
import xarray as xr
import pandas as pd

PHYSICS_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"

DEFAULT_WINDOW_DAYS = 14   # length of each window (early and late)
DEFAULT_GAP_DAYS = 0       # optional gap between end of early window and start of late window


def fetch_ocean_physics(min_lon, max_lon, min_lat, max_lat, start_date, end_date,
                         out_path="ocean_physics.nc", target_grid_size=40):
    """Single-window pull. See module docstring."""
    copernicusmarine.subset(
        dataset_id=PHYSICS_DATASET,
        variables=["uo", "vo", "thetao", "so"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=start_date,
        end_datetime=end_date,
        output_filename=out_path,
    )
    ds = xr.open_dataset(out_path)
    lat_n, lon_n = ds.dims["latitude"], ds.dims["longitude"]
    lat_factor = max(1, lat_n // target_grid_size)
    lon_factor = max(1, lon_n // target_grid_size)
    if lat_factor > 1 or lon_factor > 1:
        ds_coarse = ds.isel(
            latitude=slice(0, None, lat_factor), longitude=slice(0, None, lon_factor),
        ).load()
        ds.close()
        ds_coarse.to_netcdf(out_path)
        return ds_coarse
    return ds


def fetch_early_late_windows(min_lon, max_lon, min_lat, max_lat,
                              window_days=DEFAULT_WINDOW_DAYS, gap_days=DEFAULT_GAP_DAYS,
                              target_grid_size=40,
                              early_out="ocean_physics_early.nc",
                              late_out="ocean_physics_late.nc"):
    """
    Downloads two consecutive, non-overlapping windows over the SAME
    bounding box and grid target -- 'early' (further in the past) and
    'late' (closer to now) -- so a model trained on early-window features
    can be tested on late-window features at the same nodes.

    NOTE: because this uses the same target_grid_size logic independently
    for each window, the coarsened grids should align in practice, but
    always verify with align_grids() in graph_construction.py before
    trusting node-to-node correspondence across the two files.
    """
    now = pd.Timestamp.now().normalize()
    late_end = now
    late_start = late_end - pd.Timedelta(days=window_days)
    early_end = late_start - pd.Timedelta(days=gap_days)
    early_start = early_end - pd.Timedelta(days=window_days)

    print(f"Early window: {early_start.date()} -> {early_end.date()}")
    print(f"Late window:  {late_start.date()} -> {late_end.date()}")

    ds_early = fetch_ocean_physics(
        min_lon, max_lon, min_lat, max_lat,
        early_start.strftime("%Y-%m-%dT00:00:00"), early_end.strftime("%Y-%m-%dT00:00:00"),
        out_path=early_out, target_grid_size=target_grid_size,
    )
    ds_late = fetch_ocean_physics(
        min_lon, max_lon, min_lat, max_lat,
        late_start.strftime("%Y-%m-%dT00:00:00"), late_end.strftime("%Y-%m-%dT00:00:00"),
        out_path=late_out, target_grid_size=target_grid_size,
    )
    print(f"Early grid: {ds_early.dims['latitude']}x{ds_early.dims['longitude']}, "
          f"Late grid: {ds_late.dims['latitude']}x{ds_late.dims['longitude']}")
    if (ds_early.dims["latitude"], ds_early.dims["longitude"]) != \
       (ds_late.dims["latitude"], ds_late.dims["longitude"]):
        print("WARNING: early/late grids came back different shapes -- "
              "graph_construction.py's align_grids() will need to intersect "
              "them by coordinate before a temporal split will work.")
    return ds_early, ds_late


if __name__ == "__main__":
    # India east coast - Tamil Nadu, Andhra Pradesh, Odisha stretch
    # 1) full window (still produced, for the spatial-only splits)
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=DEFAULT_WINDOW_DAYS)
    fetch_ocean_physics(
        min_lon=78.0, max_lon=87.5, min_lat=8.0, max_lat=21.5,
        start_date=start.strftime("%Y-%m-%dT00:00:00"),
        end_date=end.strftime("%Y-%m-%dT00:00:00"),
        out_path="ocean_physics.nc",
    )
    # 2) early/late windows (for the temporal split)
    fetch_early_late_windows(min_lon=78.0, max_lon=87.5, min_lat=8.0, max_lat=21.5)
