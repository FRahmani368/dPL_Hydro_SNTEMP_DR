"""
Plot pre-computed wet-dry / warm-cold climate classification for the CONUS.

Loads the saved climate_class NetCDF (produced by an earlier compute run) and
renders a map with USA-only masking.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.ops import unary_union

try:
    from shapely import contains_xy
except ImportError:
    from shapely.vectorized import contains as contains_xy


# ---------------------------------------------------------------------------
# Configuration — edit BASE_DIR once; everything else is derived.
# ---------------------------------------------------------------------------
BASE_DIR    = Path("D:/DR")
INPUT_DIR   = BASE_DIR / "data" / "climate_classifications"
FIG_OUT_DIR = BASE_DIR / "evaluation_figures"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Run metadata (used only to locate the saved file and label the plot)
GCM       = "MIROC5"
SCENARIO  = "rcp60"
YEAR_FROM = 2006
YEAR_TO   = 2099

INPUT_NC = INPUT_DIR / f"climate_class_{GCM}_{SCENARIO}_{YEAR_FROM}_{YEAR_TO}_USA.nc"
OUT_PNG  = FIG_OUT_DIR / f"climate_regions_classification_{GCM}_{SCENARIO}_{YEAR_FROM}_{YEAR_TO}_USA.png"

# CONUS bounding box
LON_MIN, LON_MAX = -125, -66
LAT_MIN, LAT_MAX = 24, 50

CLASS_LABELS = {1: "wet-warm", 2: "wet-cold", 3: "dry-warm", 4: "dry-cold"}
CLASS_COLORS = ["#2ca25f", "#2b8cbe", "#fdae61", "#bdbdbd"]


# ---------------------------------------------------------------------------
# Load saved classification
# ---------------------------------------------------------------------------
if not INPUT_NC.exists():
    raise FileNotFoundError(f"Saved climate classification not found:\n  {INPUT_NC}")

print(f"Reading: {INPUT_NC}")
climate_class = xr.open_dataset(INPUT_NC)["climate_class"]

print("\nClass counts:")
for code, label in CLASS_LABELS.items():
    count = int((climate_class == code).sum().values)
    print(f"  {label:10s}: {count}")


# ---------------------------------------------------------------------------
# Mask everything outside the United States polygon
# ---------------------------------------------------------------------------
def mask_outside_usa(da):
    """Keep only grid-cell centers inside the United States; rest -> NaN."""
    shp = shapereader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    usa_geoms = [r.geometry for r in shapereader.Reader(shp).records()
                 if r.attributes["NAME_LONG"] == "United States"]
    usa_geom = unary_union(usa_geoms)

    lon2d, lat2d = np.meshgrid(da["lon"].values, da["lat"].values)
    inside_usa = contains_xy(usa_geom, lon2d, lat2d)
    return da.where(inside_usa)


climate_class_plot = mask_outside_usa(climate_class)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
cmap = ListedColormap(CLASS_COLORS)
cmap.set_bad("white")                       # NaN / outside-USA -> white
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
ax.set_facecolor("white")

ax.pcolormesh(
    climate_class_plot["lon"],
    climate_class_plot["lat"],
    climate_class_plot,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.35)

ax.set_title(
    f"Climate Classification for {SCENARIO}",
    fontsize=20,
)

legend_patches = [mpatches.Patch(color=c, label=CLASS_LABELS[i + 1])
                  for i, c in enumerate(CLASS_COLORS)]
ax.legend(handles=legend_patches, loc="lower left", frameon=True, fontsize=18)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

print(f"\nSaved figure: {OUT_PNG}")