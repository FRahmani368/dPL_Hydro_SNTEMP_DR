"""
Recharge model comparison: dPL (Daymet / GSWP3 forcings) vs WaterGAP2-2c GHM
vs observations.

Builds a scatter plot of simulated vs observed mean annual recharge across:
  - 0.5° CONUS grid (HUC12 upscaled and WaterGAP2 native resolution)
  - 208 HUC12 basins (high-resolution, native dPL output)
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.metrics import r2_score

from post.read_GHMs_dPLs import read_dPL_recharge_Daymet_daily


# ---------------------------------------------------------------------------
# Configuration — edit BASE_DIR once; everything else is derived.
# Use forward slashes; pathlib normalizes them per-OS.
# ---------------------------------------------------------------------------
BASE_DIR = Path("D:/DR")
# BASE_DIR = Path("/scratch/.../DR")   # uncomment on Linux/HPC

M_DIR             = BASE_DIR / "M"
DATA_DIR          = BASE_DIR / "data"
FIG_OUT_DIR       = BASE_DIR / "evaluation_figures"

GHM_WATERGAP_FILE = (DATA_DIR / "ISIMIP2a" / "GHMs" / "Watergap2_2c" / "GSWP3"
                     / "watergap2-2c_gswp3_nobc_hist_nosoc_co2_qr_global_monthly_1901_2010.nc4")
HUC12_208_FILE    = DATA_DIR / "recharge_208basins" / "huc12_recharge_208.csv"
ATTR_4231_NPY     = DATA_DIR / "ts_data_4231" / "attr_HUC12_4231_grid_clip_20250224.npy"
ATTR_4231_JSON    = DATA_DIR / "ts_data_4231" / "attr_HUC12_4231_grid_clip_20250224_name.json"
DPL_RECHARGE_DIR  = M_DIR / "daymet_1223_1023_PUB_huc12_dmt_4231"
UPSCALE_DIR       = M_DIR / "upscale"

FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# US bounding box for clipping global grids
BOUNDS_USA = dict(lon_min=-127, lat_min=20.9, lon_max=-66, lat_max=50.4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def calculate_R2(obs, sim):
    """Coefficient of determination (sklearn convention)."""
    return r2_score(obs, sim)


def rel_L2(y_sim, y_obs):
    """Relative L2 error: sqrt(sum((sim-obs)^2)) / sqrt(sum(obs^2))."""
    num = np.sqrt(np.nansum((y_sim - y_obs) ** 2))
    den = np.sqrt(np.nansum(y_obs ** 2))
    return num / den


# ---------------------------------------------------------------------------
# 1. WaterGAP2-2c (GSWP3) — monthly global recharge, clip to CONUS, 1981-2010
# ---------------------------------------------------------------------------
rech_GHM = xr.open_dataset(GHM_WATERGAP_FILE)["qr"]
rech_GHM_USA = rech_GHM.sel(
    lat=slice(BOUNDS_USA["lat_max"], BOUNDS_USA["lat_min"]),
    lon=slice(BOUNDS_USA["lon_min"], BOUNDS_USA["lon_max"]),
)

# File covers 1901-01 .. 2010-12 monthly
ghm_dates = pd.date_range(start="1901-01-01", end="2011-01-01", freq="ME")
ghm_mask  = (ghm_dates > "1980-12-31") & (ghm_dates < "2011-01-01")

# qr is kg m^-2 s^-1 → mm/month (assuming 30-day months), then to mm/year
rech_GHM_USA_monthly = rech_GHM_USA[ghm_mask, :, :] * 86400 * 30
rech_GHM_USA_yearly  = (rech_GHM_USA_monthly.sum(dim="time")
                        / np.floor(rech_GHM_USA_monthly.shape[0] / 12))


# ---------------------------------------------------------------------------
# 2. HUC12 observations (208 basins) and matching dPL daily simulations
# ---------------------------------------------------------------------------
recharge_huc12_obs_208 = pd.read_csv(HUC12_208_FILE, header=0)
obs_208     = recharge_huc12_obs_208["Groundwater recharge [mm/y]_mean"].to_numpy()
site_no_208 = recharge_huc12_obs_208["site_no"].tolist()

attr4231 = np.load(ATTR_4231_NPY)
with open(ATTR_4231_JSON, "r") as f:
    attr_name_np_4231 = json.load(f)

site_no_4231 = attr4231[:, attr_name_np_4231.index("site_no_int")].tolist()
ind_208 = [np.where(np.array(site_no_4231) == s)[0][0]
           for s in site_no_208 if s in site_no_4231]

# Daily recharge for the 208 basins (high-resolution comparison)
recharge_208 = read_dPL_recharge_Daymet_daily(
    model_name="mSS",
    start_date_mask="1980-12-31",
    end_date_mask="2010-12-31",
    site_ind_list=np.array(ind_208),
    start_date_sim="1980-01-01",
    end_date_sim="2023-01-01",
    dir0=str(DPL_RECHARGE_DIR) + "/",   # function expects trailing separator
)


# ---------------------------------------------------------------------------
# 3. dPL gridded simulations (HUC12 → 0.5°), Daymet and GSWP3 forcings
# ---------------------------------------------------------------------------
rech_dPL_huc12_dmt   = xr.open_dataset(UPSCALE_DIR / "mSS_recharge_grid_huc12_dmt_4231_1980_2023.nc")["mSS_recharge_huc12_dmt"]
rech_dPL_huc12_GSWP3 = xr.open_dataset(UPSCALE_DIR / "mSS_recharge_grid_huc12_GSWP3_4238_1962_2011.nc")["mSS_recharge_huc12_GSWP3"]
rech_obs_mean        = xr.open_dataset(UPSCALE_DIR / "obs_mean_recharge_grid.nc")["obs_recharge"]

# Daymet-forced dPL: clip to <2011 (data starts 1980-12-31, ends 2022-12-31)
dmt_dates = pd.date_range(start="1980-12-31", end="2022-12-31", freq="D")
dmt_mask  = (dmt_dates > "1972-01-01") & (dmt_dates < "2011-01-01")
rech_dPL_dmt_clip   = rech_dPL_huc12_dmt[dmt_mask, :, :]
rech_dPL_dmt_yearly = rech_dPL_dmt_clip.sum(dim="time") / np.floor(rech_dPL_dmt_clip.shape[0] / 365)

# GSWP3-forced dPL: clip to >1980
gswp3_dates = pd.date_range(start="1962-01-01", end="2010-12-31", freq="D")
gswp3_mask  = gswp3_dates > "1980-12-31"
rech_dPL_gswp_clip   = rech_dPL_huc12_GSWP3[gswp3_mask, :, :]
rech_dPL_gswp_yearly = rech_dPL_gswp_clip.sum(dim="time") / np.floor(rech_dPL_gswp_clip.shape[0] / 365)


# ---------------------------------------------------------------------------
# 4. Build scatter arrays and compute metrics
# ---------------------------------------------------------------------------
# 0.5° grid comparison: drop NaN obs cells
mask_grid     = ~np.isnan(rech_obs_mean.values.flatten())
obs_grid      = rech_obs_mean.values.flatten()[mask_grid]
ghm_grid      = rech_GHM_USA_yearly.values.flatten()[mask_grid]
dpl_gsw_grid  = rech_dPL_gswp_yearly.values.flatten()[mask_grid]
dpl_dmt_grid  = rech_dPL_dmt_yearly.values.flatten()[mask_grid]

# High-resolution comparison: 208 HUC12 basins with valid obs
mask_basin = ~np.isnan(obs_208)
obs_hi  = obs_208[mask_basin]
sim_hi  = np.nansum(recharge_208[:, mask_basin], axis=0) / np.floor(recharge_208.shape[0] / 365)

R2_watergap     = calculate_R2(obs_grid, ghm_grid)
R2_dPL_GSWP3    = calculate_R2(obs_grid, dpl_gsw_grid)
R2_dPL_dmt_low  = calculate_R2(obs_grid, dpl_dmt_grid)
R2_dPL_dmt_high = calculate_R2(obs_hi,   sim_hi)


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 13))
plt.plot([0, 800], [0, 800], lw=1.5, label="_nolegend_")
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.09)

plt.scatter(obs_grid, ghm_grid,     c="red",    s=200, marker="o")
plt.scatter(obs_grid, dpl_gsw_grid, c="black",  s=200, marker="s")
plt.scatter(obs_grid, dpl_dmt_grid, c="blue",   s=200, marker="D")
plt.scatter(obs_hi,   sim_hi,       c="orange", s=200, marker="*")

legend_labels = [
    f"Watergap2_2c(GSWP3, 0.5°)-L2: {rel_L2(ghm_grid, obs_grid):.2f}, R2: {R2_watergap:.2f}",
    f"δPS(GSWP3, HUC12 to 0.5°)-L2: {rel_L2(dpl_gsw_grid, obs_grid):.2f}, R2: {R2_dPL_GSWP3:.2f}",
    f"δPS(Daymet, HUC12 to 0.5°)-L2: {rel_L2(dpl_dmt_grid, obs_grid):.2f}, R2: {R2_dPL_dmt_low:.2f}",
    f"δPS(Daymet, HUC12)-L2: {rel_L2(sim_hi, obs_hi):.2f}, R2: {R2_dPL_dmt_high:.2f}",
]
plt.legend(legend_labels, loc="upper right", title="Models",
           fontsize=25, title_fontsize=25)

plt.xlim(-10, 800)
plt.ylim(-10, 800)
plt.grid()
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel("Recharge observations (mm/year)", fontsize=35)
plt.ylabel("Recharge simulations (mm/year)", fontsize=35)
plt.title("Observed and simulated recharge\nin δ and GHM modelings", fontsize=35)

out_path = FIG_OUT_DIR / "rech_dPL_huc12_dmt_GSWP3_hist.png"
plt.savefig(out_path, dpi=600)
print(f"Saved figure to: {out_path}")
print("END")