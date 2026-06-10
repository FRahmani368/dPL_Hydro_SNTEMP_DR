"""
Per-zone yearly recharge time series for δPS-mSS and GHM ensembles, ISIMIP2b
RCP6.0 and RCP8.5. Produces three 4x3 figures (one panel per CONUS zone):

  - plot_future_trend_recharge_mvAve.png
      Raw yearly mean (faint) + 5-yr moving-average smooth (bold).
      Units: mm/year. Four colors (one per model/scenario).

  - plot_relative_future_trend_recharge_mvAve.png
      (Smoothed - smoothed[0]) faint + 2nd-order OLS polynomial fit bold.
      Units: mm. Colors: blue=δPS, orange=GHM; solid=RCP6.0, dashed=RCP8.5.

  - perc_plot_relative_future_trend_recharge_mvAve.png
      Polynomial fit of 100*(smoothed-start)/start, shifted to start at 0.
      No faint background. Units: %. Same color scheme as the relative plot.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Configuration — edit BASE_DIR once; everything else is derived.
# ---------------------------------------------------------------------------
BASE_DIR    = Path("D:/DR")
DATA_DIR    = BASE_DIR / "data"
M_DIR       = BASE_DIR / "M"
FIG_OUT_DIR = BASE_DIR / "evaluation_figures"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARLY_NPZ     = M_DIR / "qr_2b" / "sim_recharge_dict_yearly_2007_2100.npz"
ATTR_2003_NPY  = DATA_DIR / "ts_2003basins" / "attr2003_mswep_03122024.npy"
ATTR_2003_JSON = DATA_DIR / "ts_2003basins" / "attr2003_mswep_03122024_name.json"
ZONES_SHP      = DATA_DIR / "Zones" / "Zones_0228.shp"

# Saved .npz spans 2007..2099; index 1 = 2008. We slice 2008..2099 (92 yrs).
NPZ_START_YEAR    = 2007
SLICE_START       = 2008
SLICE_END         = 2099
MOVING_AVE_WINDOW = 5


# ---------------------------------------------------------------------------
# Two model layouts:
#   MODELS_RAW  – figure 1 uses 4 distinct colors (no rcp-by-linestyle)
#   MODELS_POLY – figures 2 & 3 distinguish rcp by linestyle, model by color
# (model_key, legend_label, color, linestyle)
# ---------------------------------------------------------------------------
MODELS_RAW = [
    ("dPL_ens_mSS_rcp60_yearly", "δPS-RCP6.0", "blue",   "-"),
    ("GHM_ens_rcp60_yearly",     "GHM-RCP6.0", "red",    "-"),
    ("dPL_ens_mSS_rcp85_yearly", "δPS-RCP8.5", "black",  "-"),
    ("GHM_ens_rcp85_yearly",     "GHM-RCP8.5", "orange", "-"),
]
MODELS_POLY = [
    ("dPL_ens_mSS_rcp60_yearly", "δPS-RCP6.0", "blue",   "-"),
    ("GHM_ens_rcp60_yearly",     "GHM-RCP6.0", "orange", "-"),
    ("dPL_ens_mSS_rcp85_yearly", "δPS-RCP8.5", "blue",   "--"),
    ("GHM_ens_rcp85_yearly",     "GHM-RCP8.5", "orange", "--"),
]


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def adaptive_moving_average(arr, window):
    """Moving average with mirror-padding at the start."""
    if len(arr) < window:
        raise ValueError("Array length must be at least equal to the window size.")
    pad    = arr[:window - 1][::-1]
    padded = np.concatenate([pad, arr])
    return np.array([padded[i:i + window].mean() for i in range(len(arr))])


def fit_quadratic_ols(y, x):
    """Fit y ~ a x² + b x + c via OLS. Return fitted curve at x."""
    X   = sm.add_constant(np.column_stack((x ** 2, x)))
    res = sm.OLS(y, X).fit()
    c, a, b = res.params
    return a * x ** 2 + b * x + c


# ---------------------------------------------------------------------------
# Panel-curve generators — each returns (background, foreground) for one panel
# ---------------------------------------------------------------------------
def curves_raw_mvave(zone_mean, years):
    """Background = raw mean; foreground = moving average."""
    smoothed = adaptive_moving_average(zone_mean, MOVING_AVE_WINDOW)
    return zone_mean, smoothed


def curves_relative(zone_mean, years):
    """Background = (smoothed − smoothed[0]); foreground = quadratic fit shifted to start at 0."""
    smoothed = adaptive_moving_average(zone_mean, MOVING_AVE_WINDOW)
    y        = smoothed - smoothed[0]
    fit      = fit_quadratic_ols(y, years)
    return y, fit - fit[0]      # <-- shift polynomial to start at 0 too


def curves_percentage(zone_mean, years):
    """No background; foreground = quadratic fit of % change, shifted to 0."""
    smoothed = adaptive_moving_average(zone_mean, MOVING_AVE_WINDOW)
    y        = 100.0 * (smoothed - smoothed[0]) / smoothed[0]
    y_fit    = fit_quadratic_ols(y, years)
    return None, y_fit - y_fit[0]


# ---------------------------------------------------------------------------
# 1. Load yearly recharge .npz, slice to 2008–2099
# ---------------------------------------------------------------------------
print(f"Loading: {YEARLY_NPZ}")
loaded = np.load(YEARLY_NPZ)
i0 = SLICE_START - NPZ_START_YEAR
i1 = SLICE_END - NPZ_START_YEAR + 1

needed_keys = list({m[0] for m in (MODELS_RAW + MODELS_POLY)})
sim_yearly  = {k: np.array(loaded[k][i0:i1, :]) for k in needed_keys}
loaded.close()

years = np.arange(SLICE_START, SLICE_END + 1)


# ---------------------------------------------------------------------------
# 2. Spatial setup — basin points + zone polygons + zone→basin index map
# ---------------------------------------------------------------------------
with open(ATTR_2003_JSON, "r") as f:
    a_name = json.load(f)

attr2003 = np.load(ATTR_2003_NPY)
lat_all  = attr2003[:, a_name.index("lat")]
lon_all  = attr2003[:, a_name.index("lon")]

gdf_points = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in zip(lat_all, lon_all)],
    crs="EPSG:4326",
)
shapefile = gpd.read_file(ZONES_SHP).to_crs(gdf_points.crs)

zone_index, zone_id = {}, {}
for i, polygon_row in shapefile.iterrows():
    pgdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)
    zone_index[i] = gpd.sjoin(gdf_points, pgdf, how="inner",
                              predicate="intersects").index.tolist()
    zone_id[i] = polygon_row["new_FID"]


# ---------------------------------------------------------------------------
# 3. Shared figure builder
# ---------------------------------------------------------------------------
def make_figure(models, panel_curves, ylabel, fig_title, out_path,
                legend_panel, legend_loc="upper left",
                bg_lw=2.0, fg_lw=4.0, bg_alpha=0.4):
    fig, axs = plt.subplots(4, 3, figsize=(24, 18))
    ax = axs.flatten()
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.98, top=0.93,
                        wspace=0.17, hspace=0.24)

    for i, idx in zone_index.items():
        for model_key, label, color, linestyle in models:
            zone_mean = np.nanmean(sim_yearly[model_key][:, idx],
                                   axis=1).squeeze()
            bg, fg = panel_curves(zone_mean, years)
            leg = label if i == legend_panel else None

            if bg is not None:
                ax[i].plot(years, bg, lw=bg_lw, color=color,
                           linestyle=linestyle, alpha=bg_alpha)
            ax[i].plot(years, fg, lw=fg_lw, color=color,
                       linestyle=linestyle, label=leg)

        ax[i].tick_params(axis="both", labelsize=25)
        ax[i].set_title(f"Zone {zone_id[i]}", fontsize=28)

        if i == legend_panel:
            ax[i].legend(loc=legend_loc, fontsize=25)

    fig.text(0.5, 0.01, "Years", ha="center", fontsize=30)
    fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=30)
    fig.suptitle(fig_title, fontsize=31, y=0.985)

    plt.savefig(out_path, dpi=600)
    plt.close("all")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 4. Figure 1 — raw yearly + moving-average smooth
# ---------------------------------------------------------------------------
make_figure(
    models=MODELS_RAW,
    panel_curves=curves_raw_mvave,
    ylabel="Recharge (mm/year)",
    fig_title="Average yearly recharge and moving average recharge for different zones on GHMs and δPS",
    out_path=FIG_OUT_DIR / "plot_future_trend_recharge_mvAve.png",
    legend_panel=11,
    legend_loc="upper right",
)

# ---------------------------------------------------------------------------
# 5. Figure 2 — relative change (mm) + quadratic fit
# ---------------------------------------------------------------------------
make_figure(
    models=MODELS_POLY,
    panel_curves=curves_relative,
    ylabel="Recharge (mm)",
    fig_title="Relative recharge trend for different zones on GHMs and δPS",
    out_path=FIG_OUT_DIR / "plot_relative_future_trend_recharge_mvAve.png",
    legend_panel=8,
)

# ---------------------------------------------------------------------------
# 6. Figure 3 — percentage change (%), polynomial only, shifted to 0
# ---------------------------------------------------------------------------
make_figure(
    models=MODELS_POLY,
    panel_curves=curves_percentage,
    ylabel="Percentage of recharge (%)",
    fig_title="Percentage of average recharge changes for different zones on GHMs and δPS",
    out_path=FIG_OUT_DIR / "perc_plot_relative_future_trend_recharge_mvAve.png",
    legend_panel=8,
)

print("END")