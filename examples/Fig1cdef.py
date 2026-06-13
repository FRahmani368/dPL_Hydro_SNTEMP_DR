"""
Yearly recharge vs. precipitation scatter plots, colored by climate region
(wet/dry x warm/cold) and overlaid with per-region median curves.

ISIMIP2b ensemble, period 2008–2099. Loads pre-computed:
  - yearly recharge ensembles (δPS / GHM, rcp60 / rcp85)
  - yearly precipitation ensembles (rcp60 / rcp85)
  - climate-region labels per basin (rcp60 / rcp85)

Produces a single 2x2 figure.
"""

from pathlib import Path
import json
import pickle
import os

# Non-GUI backend (safe for batch jobs / SSH sessions without a display).
# Must be set before importing pyplot.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np


# ---------------------------------------------------------------------------
# Configuration — edit BASE_DIR once; everything else is derived.
# ---------------------------------------------------------------------------
BASE_DIR    = Path("D:/DR")
M_DIR       = BASE_DIR / "M"
DATA_DIR    = BASE_DIR / "data"
FIG_OUT_DIR = BASE_DIR / "evaluation_figures"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

RECHARGE_YEARLY_NPZ = M_DIR    / "qr_2b" / "sim_recharge_dict_yearly_2007_2100.npz"
PRECIP_PKL          = DATA_DIR / "ISIMIP2b" / "precip_dict.pkl"
LABELS_RCP60_JSON   = DATA_DIR / "climate_classifications" / "2b_climate_labels_rcp60.json"
LABELS_RCP85_JSON   = DATA_DIR / "climate_classifications" / "2b_climate_labels_rcp85.json"

# Period of interest — this run only produces the 2008-2099 figure.
# The yearly NPZ spans 2007-2100; slice [1:93] gives years 2008..2099 (92 yrs).
YEAR_FROM, YEAR_TO = 2008, 2099
TREND_KEY = f"{YEAR_FROM} - {YEAR_TO}"


# ---------------------------------------------------------------------------
# Climate-region color scheme + label canonicalization
# ---------------------------------------------------------------------------
REGION_COLORS = {
    "wet-cold": "#56B4E9",  # sky blue
    "wet-warm": "#009E73",  # green
    "dry-warm": "#E69F00",  # orange
    "dry-cold": "#CC79A7",  # magenta
}
REGION_ORDER = ["wet-cold", "wet-warm", "dry-warm", "dry-cold"]

# Accept synonyms / sloppy casing in label strings
CANON = {
    "cold-wet": "wet-cold", "warm-wet": "wet-warm",
    "warm-dry": "dry-warm", "cold-dry": "dry-cold",
    "wet-cold": "wet-cold", "wet-warm": "wet-warm",
    "dry-warm": "dry-warm", "dry-cold": "dry-cold",
}


def colorize(labels_array):
    """Map an array of climate-region label strings to hex colors."""
    out = []
    for lab in labels_array:
        key = CANON.get(str(lab).strip().lower())
        out.append(REGION_COLORS.get(key, "#000000"))  # black for unknown
    return np.array(out)


LEGEND_HANDLES = [
    Line2D([0], [0], marker="o", linestyle="",
           color=REGION_COLORS[r],
           label=r.replace("-", "–").title(),  # e.g. "wet-cold" -> "Wet–Cold"
           markersize=14)
    for r in REGION_ORDER
]


# ---------------------------------------------------------------------------
# Median-curve helper (median of y in quantile bins of x)
# ---------------------------------------------------------------------------
def median_curve(x, y, nbins=40, min_bin=15):
    """
    Robust 'median regression' line:
      - bin x by quantiles, take median of y within each bin
      - skip bins with fewer than `min_bin` points
      - light 3-point running-median smoothing on the resulting curve
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return np.array([]), np.array([])

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, nbins + 1)))
    if edges.size < 2:
        return np.array([]), np.array([])

    xmed, ymed = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() >= min_bin:
            xmed.append(np.median(x[sel]))
            ymed.append(np.median(y[sel]))
    xmed, ymed = np.asarray(xmed), np.asarray(ymed)

    if xmed.size >= 5:
        order = np.argsort(xmed)
        xmed, ymed = xmed[order], ymed[order]
        ypad = np.pad(ymed, (1, 1), mode="edge")
        ymed = np.median(np.stack([ypad[:-2], ypad[1:-1], ypad[2:]], axis=1),
                         axis=1)

    return xmed, ymed


# ---------------------------------------------------------------------------
# 1. Load yearly recharge ensembles, slice to 2008–2099
# ---------------------------------------------------------------------------
print(f"Loading yearly recharge: {RECHARGE_YEARLY_NPZ}")
loaded = np.load(RECHARGE_YEARLY_NPZ)

MODELS = [
    "dPL_ens_mSS_rcp60_yearly",
    "GHM_ens_rcp60_yearly",
    "dPL_ens_mSS_rcp85_yearly",
    "GHM_ens_rcp85_yearly",
]
# slice [1:93] -> years 2008..2099 inclusive
sim_rech = {m: np.array(loaded[m][1:93, :]) for m in MODELS}
loaded.close()


# ---------------------------------------------------------------------------
# 2. Load yearly precipitation ensembles for the 2008–2099 period
# ---------------------------------------------------------------------------
print(f"Loading precipitation: {PRECIP_PKL}")
with open(PRECIP_PKL, "rb") as f:
    precip_dict = pickle.load(f)

precip_60 = precip_dict[TREND_KEY]["precip_rcp60_ens_yearly"]
precip_85 = precip_dict[TREND_KEY]["precip_rcp85_ens_yearly"]
precip_dict = None


# ---------------------------------------------------------------------------
# 3. Load climate-region labels and colorize per basin
# ---------------------------------------------------------------------------
def load_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return np.array(json.load(f)["labels_Ep_over_P"])

colors_rcp60 = colorize(load_labels(LABELS_RCP60_JSON))
colors_rcp85 = colorize(load_labels(LABELS_RCP85_JSON))


# ---------------------------------------------------------------------------
# 4. Build the 2x2 figure
# ---------------------------------------------------------------------------
PANEL_TITLES = [
    "a) δPS in RCP6.0",
    "b) GHMs ensemble in RCP6.0",
    "c) δPS in RCP8.5",
    "d) GHMs ensemble in RCP8.5",
]
y_arrays     = [sim_rech[m] for m in MODELS]          # each shape (T, L)
x_arrays     = [precip_60, precip_60, precip_85, precip_85]
point_colors = [colors_rcp60, colors_rcp60, colors_rcp85, colors_rcp85]

MAX_VAL = 3800   # axis limit (mm/year)
AX_FS   = 28     # axis font size

fig, axs = plt.subplots(2, 2, figsize=(24, 24))
plt.subplots_adjust(left=0.1, bottom=0.08, right=0.98, top=0.9,
                    wspace=0.2, hspace=0.2)
ax = axs.flatten()

for j in range(4):
    x  = x_arrays[j]
    y  = y_arrays[j]
    pc = point_colors[j]
    T  = x.shape[0]

    # Flatten valid (finite) points in matched order
    m2d = np.isfinite(x) & np.isfinite(y)
    xf  = x[m2d]
    yf  = y[m2d]
    cf  = np.tile(pc, (T, 1))[m2d]   # per-basin color repeated across time

    ax[j].scatter(xf, yf, s=12, alpha=0.8, c=cf)
    ax[j].plot([0, MAX_VAL], [0, MAX_VAL], color="lightblue", lw=1)
    ax[j].set_xlim(0, MAX_VAL)
    ax[j].set_ylim(0, MAX_VAL)
    ax[j].set_title(PANEL_TITLES[j], fontsize=AX_FS)
    ax[j].set_xlabel("Precipitation (mm/year)", fontsize=AX_FS)
    ax[j].set_ylabel("Groundwater recharge (mm/year)", fontsize=AX_FS)
    ax[j].tick_params(axis="both", labelsize=AX_FS)
    ax[j].grid(True, ls=":")

    # Median curve per climate region
    for region in REGION_ORDER:
        col = REGION_COLORS[region]
        sel = (cf == col)
        if not np.any(sel):
            continue
        xs, ys = median_curve(xf[sel], yf[sel], nbins=40, min_bin=15)
        if xs.size:
            ax[j].plot(xs, ys, color="white", lw=7, alpha=0.5, zorder=3)
            ax[j].plot(xs, ys, color=col,    lw=4, alpha=0.95, zorder=4)

    if j == 0:
        ax[j].legend(handles=LEGEND_HANDLES, fontsize=AX_FS - 4,
                     frameon=True, loc="upper left")

fig.text(0.08, 0.96,
         f"Recharge and precipitation relationships in different models in {TREND_KEY}",
         fontsize=40, ha="left", va="top")

out_path = FIG_OUT_DIR / f"rech_precip_{YEAR_FROM}_{YEAR_TO}.png"
plt.savefig(out_path, dpi=300)
plt.close("all")
print(f"Saved figure: {out_path}")
print("END")