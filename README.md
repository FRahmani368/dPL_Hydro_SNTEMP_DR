# dPL_Hydro_SNTEMP

A **differentiable programming (dPL) framework** that integrates process-based hydrological models and stream temperature models in a unified, end-to-end trainable architecture. Neural networks (LSTM or MLP) learn to predict spatially varying physical parameters across many river basins simultaneously, while differentiable physics models enforce process-based constraints — combining the interpretability of mechanistic modeling with the predictive power of deep learning.

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Configuration File In Depth](#4-configuration-file-in-depth)
   - [4.1 Paths](#41-paths)
   - [4.2 Training Control](#42-training-control)
   - [4.3 Neural Network Settings](#43-neural-network-settings)
   - [4.4 Hydrological Model Settings](#44-hydrological-model-settings)
   - [4.5 Stream Temperature Model Settings](#45-stream-temperature-model-settings)
   - [4.6 Loss Function Settings](#46-loss-function-settings)
5. [Available Hydrological Models](#5-available-hydrological-models)
6. [Available Stream Temperature Models](#6-available-stream-temperature-models)
7. [Loss Functions Explained](#7-loss-functions-explained)
8. [Input Data Format](#8-input-data-format)
9. [Running the Model](#9-running-the-model)
   - [9.1 Hydrology Only](#91-hydrology-only-no-temperature)
   - [9.2 Integrated Hydrology + Temperature](#92-integrated-hydrology--temperature)
   - [9.3 Temperature Only (Pre-saved Flows)](#93-temperature-only-with-pre-saved-flows)
10. [Understanding the Training Output](#10-understanding-the-training-output)
11. [Output Files](#11-output-files)
12. [Tips and Common Issues](#12-tips-and-common-issues)
13. [Reproducing the Data-Release Figures](#13-reproducing-the-data-release-figures)
    - [13.1 What the Scripts Do](#131-what-the-scripts-do)
    - [13.2 How to Run a Figure Script](#132-how-to-run-a-figure-script)
    - [13.3 Setting Up the Data Directory](#133-setting-up-the-data-directory)
    - [13.4 Script-by-Script Reference](#134-script-by-script-reference)
    - [13.5 Common Issues](#135-common-issues)

---

## 1. Scientific Background

### What is Differentiable Parameter Learning?

Traditional hydrological models are calibrated one basin at a time by searching for the best-fitting parameters using optimization algorithms. This approach does not transfer knowledge between basins and cannot easily leverage large datasets.

This framework uses **differentiable parameter learning (dPL)**: a neural network reads basin attributes (soil type, elevation, vegetation, etc.) and meteorological forcings, and outputs the parameters of a physics-based model — for every basin at once. Because both the neural network and the physics model are implemented in PyTorch, gradients flow backward through the entire chain, allowing the system to learn which basin characteristics drive which physical parameters.

```
Basin attributes  ──┐
                     ├──► LSTM/MLP ──► Physical parameters ──► Hydro model ──► Streamflow
Meteorological    ──┘                                      └──► Temp model ──► Temperature
forcings
```

### Why Integrate Hydrology and Temperature?

Stream temperature is driven by the water that enters the stream. Groundwater (slow, deep) is cold year-round; surface runoff (fast, shallow) rapidly equilibrates with air temperature. A physically correct temperature model must know how much of the streamflow comes from each source — which requires running a hydrology model first. This framework models both simultaneously, sharing parameters and training against both streamflow and temperature observations at once.

---

## 2. Repository Structure

```
dPL_Hydro_SNTEMP/
│
├── main_hydro_temp.py              # ← Entry point. Run this script to train or test.
│
├── config/
│   ├── config_hydro_temp.yaml      # ← All configuration parameters (edit this file)
│   └── read_configurations.py      # YAML parser — loads config into a Python dict
│
├── MODELS/
│   ├── Differentiable_models.py    # Assembles NN + hydro model + temp model into one
│   │                               # differentiable PyTorch module
│   ├── train_test.py               # Training loop and testing loop
│   │
│   ├── hydro_models/               # Differentiable hydrological models
│   │   ├── HBV/                    # HBV model
│   │   ├── HBV_capillary/          # HBV with capillary rise
│   │   └── marrmot_PRMS/           # PRMS (Precipitation-Runoff Modeling System)
│   │
│   ├── temp_models/
│   │   ├── SNTEMP/                 # Energy-balance stream temperature model
│   │   └── SNTEMP_with_gw0/        # SNTEMP with 4 distinct groundwater components
│   │
│   ├── NN_models/
│   │   ├── LSTM_models.py          # CuDNN LSTM with recurrent dropout
│   │   ├── MLP_models.py           # Multi-layer perceptron
│   │   └── dropout.py              # Recurrent-dropout helper used by the LSTM
│   │
│   ├── loss_functions/             # One file per loss function
│   │   ├── RmseLoss_flow_comb.py   # Flow only (linear + log-sqrt RMSE)
│   │   ├── RmseLoss_flow_temp.py   # Flow + temperature
│   │   ├── NSEsqrtLoss_flow_temp.py# Scale-normalized NSE for flow + temperature
│   │   ├── crit.py                 # Shared loss/metric primitives
│   │   └── get_loss_function.py    # Factory that loads the correct loss class
│   │
│   └── PET_models/                 # Potential evapotranspiration methods
│       └── potet.py
│
├── core/
│   ├── load_data/
│   │   ├── dataFrame_loading.py    # Reads NPY / Feather / CSV forcing and attribute files
│   │   ├── data_prep.py            # Mini-batch sampling, train/test splitting
│   │   ├── normalizing.py          # Computes and applies z-score normalization
│   │   └── time.py                 # YYYYMMDD date range utilities
│   └── utils/
│       ├── small_codes.py          # Output directory creation, source flow unit conversion
│       ├── randomseed_config.py    # Reproducibility seed management
│       └── grid.py                 # Grid-related helpers
│
├── post/                           # Post-processing: statistics (NSE, KGE) and plots
├── nonlinearSolver/                # Implicit numerical solvers for stiff ODEs
├── HydroModels_no_use/             # Archived / deprecated model variants
│
├── pyproject.toml                  # Python dependencies (managed with uv)
└── ruff.toml                       # Code style configuration
```

---

## 3. Environment Setup

This project uses [**uv**](https://docs.astral.sh/uv/) for dependency management — it is significantly faster than pip/conda and produces fully reproducible environments through a lockfile.

### Step 1 — Install uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, close and reopen your terminal so the `uv` command is available.

### Step 2 — Clone the repository

```bash
git clone https://github.com/your-org/dPL_Hydro_SNTEMP.git
cd dPL_Hydro_SNTEMP
```

### Step 3 — Create the virtual environment and install all dependencies

```bash
uv sync
```

This single command:
- Creates a `.venv` folder in the project directory
- Installs the exact package versions from `uv.lock`
- Downloads PyTorch with **CUDA 12.8** support (required for RTX 4000/5000 series GPUs)

> **GPU requirements:** An NVIDIA GPU is required. The default configuration targets CUDA 12.8 (`cu128`), which supports Blackwell (RTX 5090, sm_120), Ada Lovelace (RTX 4090), and Ampere (RTX 3090) architectures. If you have a different GPU generation, update the index URL in `pyproject.toml`:
>
> ```toml
> [[tool.uv.index]]
> name = "pytorch-cu128"
> url = "https://download.pytorch.org/whl/cu128"   # change to cu126, cu121, cu118, etc.
> ```
>
> After changing the URL, run `uv sync` again to reinstall.

### Step 4 — Activate the environment

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your terminal prompt.

### Step 5 — Verify the GPU is accessible

```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output (example):
```
True
NVIDIA GeForce RTX 5090
```

If `torch.cuda.is_available()` returns `False`, your PyTorch build does not match your CUDA version. Check the CUDA version installed on your system with `nvidia-smi` and update the index URL accordingly.

---

## 4. Configuration File In Depth

All model behavior is controlled through a single YAML file:

```
config/config_hydro_temp.yaml
```

You never need to edit Python source files for typical experiments — only this configuration file. Below is a complete explanation of every parameter.

---

### 4.1 Paths

```yaml
forcing_path: "D:\\data\\forcing_415.npy"
attr_path:    "D:\\data\\attr_415.npy"
output_model: "D:\\models\\experiment_01"
```

| Parameter | What it points to |
|-----------|-------------------|
| `forcing_path` | Time-varying meteorological inputs (precipitation, temperature, PET, etc.) stored as a NumPy `.npy` or Feather `.feather` file |
| `attr_path` | Static basin characteristics (soil type, elevation, slope, vegetation, etc.) |
| `output_model` | Root directory where trained model checkpoints and prediction files will be written. Sub-folders are created automatically with timestamps. |

> **Multiple machine support:** The config file has commented-out blocks for different server environments (ICDS cluster, Windows workstation, Confucius server). You can keep all paths in the file and comment/uncomment the block that matches your current machine.

---

### 4.2 Training Control

```yaml
randomseed: [0]
Action: [0, 1]
device: "cuda"
nmul: 2

tRange:   [19800101, 20230101]
t_train:  [20180101, 20230101]
t_test:   [19990101, 20230101]
warm_up:  365

rho:        365
batch_size: 100
EPOCHS:     100
saveEpoch:  10
EPOCH_testing: 100
no_basins:  25
```

**`randomseed`** — Integer seed for NumPy and PyTorch random number generators. Setting this ensures that two runs with identical config produce identical results. Set to `[0]` for reproducibility, or `[None]` to use a random seed each run.

**`Action`** — Controls what the script does when you run it:
- `[0]` → Train only
- `[1]` → Test only (loads a saved checkpoint)
- `[0, 1]` → Train, then immediately test

**`device`** — Which hardware to use. `"cuda"` uses the first available GPU. To use a specific GPU in a multi-GPU system use `"cuda:1"` etc. Use `"cpu"` only for debugging (training will be extremely slow).

**`nmul`** — Number of parallel parameter sets estimated per basin. When `nmul=2`, the NN generates two sets of physical parameters and the model runs twice in parallel, averaging the outputs. This acts as an ensemble within a single forward pass and tends to improve robustness.

**`tRange`** — The full date range covered by your input data files. The loader uses this to index into the arrays correctly.

**`t_train`** — The period used for training. The model reads data only from this window during training. Note: the first `warm_up` days of this period are used to spin up model states and are not counted in the loss.

**`t_test`** — The period used for evaluation after training. Should not overlap with `t_train` for an honest out-of-sample test (though `warm_up` days at the start of the test period are still needed for spin-up).

**`warm_up`** — Number of days at the start of each sequence used to initialize model states (soil moisture, groundwater storage, snowpack, etc.) before loss is computed. **365 days (1 year) is recommended** for most hydro models to avoid unrealistic initial conditions influencing the results.

**`rho`** — The length of each training sequence in days. The LSTM processes one contiguous window of `rho` days per mini-batch sample. Longer sequences capture more temporal structure but use more GPU memory.

**`batch_size`** — Number of basins sampled per mini-batch iteration. Each iteration randomly picks `batch_size` basins and a random starting day, runs the model for `rho` days on those basins, computes the loss, and updates weights.

**`EPOCHS`** — Total number of full passes through the training data.

**`saveEpoch`** — A model checkpoint (`model_EpN.pt`) is saved every N epochs. For example, `saveEpoch: 10` with `EPOCHS: 100` saves 10 checkpoints.

**`EPOCH_testing`** — When `Action` includes `1` (test), this specifies which saved checkpoint to load. Set this to the epoch with the best validation performance.

**`no_basins`** — Number of basins to evaluate during testing. Useful for quick evaluation on a subset.

---

### 4.3 Neural Network Settings

```yaml
NN_model_name: "LSTM"
hidden_size:   256
dropout:       0.5

varT_NN: ['prcp(mm/day)', 'tmean(C)', 'PET_hargreaves(mm/day)']

varC_NN: [
  'aridity', 'p_mean', 'ETPOT_Hargr', 'NDVI', 'FW', 'SLOPE_PCT',
  'SoilGrids1km_sand', 'SoilGrids1km_clay', 'SoilGrids1km_silt',
  'glaciers', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt',
  'ELEV_MEAN_M_BASIN', 'meanTa', 'permafrost', 'permeability',
  'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction',
  'T_clay', 'T_gravel', 'T_sand', 'T_silt', 'Porosity', 'DRAIN_SQKM'
]
```

**`NN_model_name`** — Which neural network architecture to use as the parameter estimator:
- `"LSTM"` *(recommended)* — A Long Short-Term Memory recurrent network. It processes the time-varying forcing sequence day by day and maintains a hidden state that captures memory of past conditions (antecedent wetness, snowpack buildup, etc.). Because hydrology has strong memory effects (e.g., groundwater recharge from storms weeks ago), LSTM is well-suited.
- `"MLP"` — A feed-forward Multi-Layer Perceptron. It cannot learn temporal patterns by itself, so it is better suited when only static attributes are important, or when you want a simpler, faster baseline.

**`hidden_size`** — The number of units in the LSTM's hidden state (or the width of MLP hidden layers). Larger values give the network more capacity to represent complex relationships between basin attributes and parameters, but require more memory and are slower to train. **256 is a reasonable default.** Values of 128–512 are typical.

**`dropout`** — Dropout probability applied to the LSTM's recurrent weight matrices during training. This is a regularization technique that randomly zeros out connections to prevent overfitting. `0.5` means 50% of connections are dropped each forward pass during training (at test time, dropout is disabled). For small datasets (few basins), increase toward `0.7`. For very large datasets, you can decrease to `0.3`.

**`varT_NN`** — The list of **time-varying** (daily) variables fed to the neural network as input. These come from your forcing file. The LSTM sees these as a sequence, so the order of variables in the list does not matter — what matters is that the variable names exactly match the column names in your forcing data.

Common choices:
- `'prcp(mm/day)'` — Daily precipitation (always include)
- `'tmean(C)'` — Mean daily air temperature (always include)
- `'PET_hargreaves(mm/day)'` — Potential evapotranspiration estimated by Hargreaves method
- `'srad(W/m2)'` — Solar radiation (useful if running temperature model)
- `'vp(Pa)'` — Vapor pressure
- `'tmin(C)'`, `'tmax(C)'` — Min/max temperature

**`varC_NN`** — The list of **static** basin attributes fed to the neural network. These do not change with time — they describe the physical character of each basin. They are repeated at every time step and concatenated with `varT_NN`. The more informative attributes you include, the better the NN can differentiate basins and estimate appropriate parameters. Variable names must exactly match column headers in your attribute file.

The config file contains three commented-out attribute sets for different datasets (MSWEP/GAGES-II, CAMELS, and Yalan's CAMELS). **Uncomment the block that matches your dataset** and comment out the others.

---

### 4.4 Hydrological Model Settings

```yaml
hydro_model_name: "marrmot_PRMS"

varT_hydro_model: ['prcp(mm/day)', 'tmean(C)', 'PET_hargreaves(mm/day)']
varC_hydro_model: ['DRAIN_SQKM']

routing_hydro_model: True
dyn_params_list_hydro: []
dydrop: 0.0

# Only needed if hydro_model_name is "None"
flow_data_path: "path/to/precomputed_flow.npy"
```

**`hydro_model_name`** — Which process-based hydrology model to use. See [Section 5](#5-available-hydrological-models) for a detailed description of each. Set to `"None"` if you want to skip hydrology and run the temperature model using pre-computed flows from a file (see `flow_data_path`).

**`varT_hydro_model`** — Time-varying variables passed directly to the hydrology model's equations (separate from `varT_NN`). For most models this is precipitation, temperature, and PET. These are the physical forcing inputs to the differential equations — not the NN inputs.

**`varC_hydro_model`** — Static attributes passed directly to the hydrology model. Most models only need the basin drainage area (`'DRAIN_SQKM'` or `'area_gages2'`) for unit conversion from mm/day to m³/s.

**`routing_hydro_model`** — Whether to apply convolution-based streamflow routing after the hydrology model. When `True`, the NN also estimates two routing parameters (`a` and `b`) that define a gamma-distribution unit hydrograph. This smooths and delays the simulated hydrograph, representing the travel time of water through the river channel. **Recommended: `True`** for most applications, especially larger basins where travel time is significant.

**`dyn_params_list_hydro`** — A list of parameter names that should be **time-varying** (estimated at every time step) rather than static per-basin. For example:
```yaml
dyn_params_list_hydro: ["parBETA", "parK0"]
```
This tells the model that soil drainage parameters vary seasonally (e.g., due to freezing/thawing or vegetation activity). An empty list `[]` means all parameters are static scalars estimated once per basin — which is the standard dPL approach and a good starting point.

**`dydrop`** — Dropout rate applied specifically to dynamic parameters. When `dyn_params_list_hydro` is non-empty, setting `dydrop > 0` randomly zeros some dynamic parameter estimates during training. Leave at `0.0` for standard training.

**`flow_data_path`** — Only used when `hydro_model_name: "None"`. Points to a pre-computed flow simulation (e.g., from a previous training run) that will be used as input to the temperature model instead of running a hydrology model online.

---

### 4.5 Stream Temperature Model Settings

```yaml
temp_model_name: "SNTEMP"
routing_temp_model: True

res_time_lenF_srflow:     1
res_time_lenF_ssflow:     30
res_time_lenF_bas_shallow: 180
res_time_lenF_gwflow:     365

lat_temp_adj: True

# Physical constants (rarely need changing)
STemp_default_emissivity_veg: 0
STemp_default_albedo:         0.1
STemp_default_delta_Z:        1.0
params_water_density:         1000
params_C_w:                   4184
NEARZERO:                     1e-3
Epan_coef:                    1.0
initial_values_shade_fraction: 0.1

varT_temp_model: ["ccov", "vp(Pa)", "srad(W/m2)"]
varC_temp_model: ["DRAIN_SQKM", "stream_length_square", "SLOPE_PCT", "ELEV_MEAN_M_BASIN"]

dyn_params_list_temp: []
res_time_type: "SNTEMP"
```

**`temp_model_name`** — Which stream temperature model to use:
- `"None"` — No temperature model. The framework runs hydrology only.
- `"SNTEMP"` — The full SNTEMP energy balance model. Requires the hydrology model to be running (i.e., `hydro_model_name` cannot also be `"None"`).
- `"SNTEMP_gw0"` — A variant of SNTEMP that tracks 4 distinct groundwater outflow components separately, allowing each to enter the stream at a different temperature.

**`routing_temp_model`** — Each flow source (surface runoff, subsurface interflow, groundwater) has a different characteristic temperature depending on how long it has been in contact with the subsurface. When `True`, a separate convolution filter is applied to each flow source before it enters the temperature model. This means a rain event does not instantly heat the stream — the effect is spread over multiple days according to the residence time. **Always set to `True` when running SNTEMP.**

**`res_time_lenF_srflow`** — Residence time filter length (in days) for **surface runoff**. Surface runoff moves through the stream quickly, so this is short (1–5 days). Water temperature for surface flow is close to air temperature.

**`res_time_lenF_ssflow`** — Residence time filter length for **subsurface interflow** (shallow soil water). Slower than surface runoff, typical range 10–60 days. This water has been moderately buffered by the soil.

**`res_time_lenF_bas_shallow`** — Residence time filter length for **shallow baseflow**. Water from the shallow groundwater system, typical range 60–200 days. Temperature is buffered significantly.

**`res_time_lenF_gwflow`** — Residence time filter length for **deep groundwater baseflow**. Deep groundwater moves very slowly and maintains a near-constant temperature close to the mean annual air temperature. Typical range 200–500 days.

**`lat_temp_adj`** — Whether to include a latitude-based adjustment to the solar radiation received at each basin. When `True`, the NN estimates an additional adjustment parameter, allowing the model to correct for the effects of aspect and topographic shading that are not captured by the regional solar radiation forcing.

**Physical constants** (rarely need changing):
- `STemp_default_emissivity_veg` — Emissivity of vegetation for longwave radiation calculations
- `STemp_default_albedo` — Default water surface albedo (fraction of solar radiation reflected)
- `STemp_default_delta_Z` — Streambed depth for conductive heat exchange (meters)
- `params_water_density` — Water density (kg/m³)
- `params_C_w` — Specific heat capacity of water (J/kg·K)
- `Epan_coef` — Pan evaporation coefficient for latent heat calculations
- `initial_values_shade_fraction` — Starting value for the riparian shade fraction parameter

**`varT_temp_model`** — Time-varying variables passed directly to the temperature model's energy balance equations:
- `"ccov"` — Cloud cover fraction (0–1). Controls how much solar radiation reaches the stream surface.
- `"vp(Pa)"` — Vapor pressure. Used for latent heat (evaporative cooling) calculations.
- `"srad(W/m2)"` — Incoming solar radiation. Primary heat input driver.

**`varC_temp_model`** — Static attributes passed to the temperature model:
- `"DRAIN_SQKM"` or `"area_gages2"` — Basin drainage area, for flow unit conversion
- `"stream_length_square"` — Stream channel length within the basin
- `"SLOPE_PCT"` — Channel slope, affects flow velocity and thus residence time
- `"ELEV_MEAN_M_BASIN"` — Mean basin elevation

**`dyn_params_list_temp`** — Same concept as `dyn_params_list_hydro` but for temperature model parameters. For example, riparian shade fraction varies seasonally (leaves in summer, bare in winter). To make shade dynamic:
```yaml
dyn_params_list_temp: ["w1_shade"]
```

**`res_time_type`** — Algorithm used to compute the convolution filter shape:
- `"SNTEMP"` *(recommended)* — Uses the gamma-distribution parameterization from the original SNTEMP model
- `"van Vliet"` — Alternative parameterization from van Vliet et al.
- `"Meisner"` — Alternative parameterization from Meisner et al.

---

### 4.6 Loss Function Settings

```yaml
loss_function: "RmseLoss_flow_comb"
loss_function_weights:
    w1: 3.0
    w2: 1.0

target: ["00060_Mean"]
```

**`loss_function`** — Name of the loss function class to use. Must exactly match one of the filenames in `MODELS/loss_functions/` (without `.py`). See [Section 7](#7-loss-functions-explained) for a full description of each option.

**`loss_function_weights`** — Relative weights between the components of multi-objective loss functions:
- `w1` — Weight on the streamflow component
- `w2` — Weight on the temperature component

These are **not** required to sum to 1. A ratio of `w1=3, w2=1` means streamflow errors are penalized three times more than temperature errors. Adjust these based on the relative importance of each target and the scale of the errors.

**`target`** — The list of observation variables used as training targets. Must match the column order in your observation data file. Only include variables that your chosen loss function requires:

| Variable name | What it represents | Units |
|---|---|---|
| `"00060_Mean"` | Daily mean streamflow | m³/s |
| `"00010_Mean"` | Daily mean stream temperature | °C |

> The loss functions shipped in this release (see [Section 7](#7-loss-functions-explained)) train on streamflow and stream temperature only. Use `target: ["00060_Mean"]` for flow-only training and `target: ["00060_Mean", "00010_Mean"]` for coupled flow + temperature training.

---

## 5. Available Hydrological Models

All models are **fully differentiable PyTorch implementations** and can be used interchangeably by changing `hydro_model_name` in the config. Each model internally defines the physical bounds for its parameters — the neural network outputs are automatically scaled to these bounds via a sigmoid transformation.

### HBV (`"HBV"`)

The **Hydrologiska Byråns Vattenbalansmodell** — a widely used conceptual bucket model from Sweden. It simulates snow accumulation/melt, soil moisture, and a two-reservoir groundwater system. It is well-tested globally and a good general-purpose starting point.

| Parameter | Range | Description |
|-----------|-------|-------------|
| `parBETA` | 1–6 | Nonlinearity of recharge from soil to groundwater |
| `parFC` | 50–1000 mm | Field capacity of the soil moisture store |
| `parK0` | 0.05–0.9 d⁻¹ | Fast runoff recession coefficient (upper zone) |
| `parK1` | 0.01–0.5 d⁻¹ | Slow runoff recession coefficient (lower zone) |
| `parK2` | 0.001–0.2 d⁻¹ | Baseflow recession coefficient |
| `parLP` | 0.2–1 | Fraction of FC above which AET = PET |
| `parPERC` | 0–10 mm/d | Maximum percolation from upper to lower zone |
| `parUZL` | 0–100 mm | Threshold for fast runoff generation |
| `parTT` | −2.5–2.5 °C | Temperature threshold for snow vs. rain |
| `parCFMAX` | 0.5–10 mm/°C/d | Degree-day snowmelt factor |
| `parCFR` | 0–0.1 | Refreezing coefficient |
| `parCWH` | 0–0.2 | Fraction of snowpack that can hold liquid water |

Flow outputs: `flow_sim` (total), `ssflow` (subsurface), `gwflow` (groundwater baseflow).

### HBV with Capillary Rise (`"HBV_capillary"`)

An extension of HBV that adds a capillary rise flux from the lower groundwater zone back up to the soil moisture store during dry periods. Useful in regions with shallow water tables where capillary suction is significant.

### PRMS (`"marrmot_PRMS"`)

The **Precipitation-Runoff Modeling System** from the USGS, ported from the MARRMoT (Modular Assessment of Rainfall-Runoff Models Toolbox). A flexible, multi-process model with interception, snow, soil, interflow, groundwater, and deep sink components.

| Parameter | Range | Description |
|-----------|-------|-------------|
| `tt` | −3–5 °C | Temperature threshold for snow/rain |
| `ddf` | 0–20 mm/°C/d | Degree-day snowmelt factor |
| `alpha` | 0–1 | Fraction of rainfall going to interception |
| `beta` | 0–1 | Fraction of catchment contributing to soil recharge |
| `stor` | 0–5 mm | Maximum interception capacity |
| `scx` | 0–1 | Maximum contributing area for saturation excess flow |
| `flz` | 0.005–0.995 | Fraction of total soil that is lower zone |
| `stot` | 1–2000 mm | Total soil moisture storage |
| `cgw` | 0–20 mm/d | Constant drainage to deep groundwater |
| `k1`–`k6` | 0–1 | Various recession and drainage coefficients |

---

## 6. Available Stream Temperature Models

### SNTEMP (`"SNTEMP"`)

A **physically-based energy balance** model that computes daily mean stream temperature by accounting for all major heat exchange processes between the stream and its environment:

1. **Solar shortwave radiation** — Direct sunlight heats the water. Modulated by cloud cover and a learned riparian shade fraction (fraction of the stream surface shaded by trees or topography).
2. **Atmospheric longwave radiation** — The atmosphere emits heat toward the stream. Driven by air temperature and cloud cover.
3. **Latent heat (evaporative cooling)** — Water evaporating from the stream surface cools it. Driven by vapor pressure deficit and wind.
4. **Streambed conduction** — Heat conducted between the streambed sediment and the water. Driven by the temperature difference between streambed and water.
5. **Advective heat from inflows** — The most important coupling with the hydrology model. Each water source (surface runoff, subsurface interflow, groundwater) enters at a different temperature:
   - **Surface runoff** enters near air temperature (residence time 1–5 days)
   - **Subsurface interflow** is buffered by travel through the shallow soil (residence time weeks)
   - **Groundwater** is buffered by travel through deep aquifers (residence time months to years) and enters near the mean annual air temperature

The convolution filters for each flow source model this buffering effect mathematically, so the stream temperature responds to a rainfall event gradually rather than instantaneously.

### SNTEMP with 4 Groundwater Components (`"SNTEMP_gw0"`)

An enhanced version of SNTEMP that tracks 4 distinct groundwater outflow pathways (e.g., primary baseflow, secondary baseflow, shallow baseflow, and a fourth component). Each component has its own residence time filter and temperature. Use this when working with a hydrology model that explicitly simulates multiple groundwater reservoirs.

---

## 7. Loss Functions Explained

The loss function determines what the model is trained to match. All loss functions handle missing observations (NaN values) automatically — basins or days with no measurements are masked out and do not contribute to the loss.

### `RmseLoss_flow_comb` ← **Default and recommended for flow-only training**

Combines two RMSE terms for streamflow:
- **Linear RMSE** — penalizes errors on high flows proportionally. Ensures peak flows are simulated well.
- **Log-sqrt RMSE** — penalizes errors on low flows proportionally. Ensures the model also simulates baseflows and droughts well.

The combined loss is: `L = (1 - alpha) * RMSE_linear + alpha * RMSE_log_sqrt`

With the default `alpha=0.25`, the linear RMSE dominates but low flows still receive meaningful attention. This balance is important because raw RMSE alone ignores low-flow errors.

**When to use:** Any experiment where you only have streamflow observations and do not need temperature.

**Config:**
```yaml
loss_function: "RmseLoss_flow_comb"
target: ["00060_Mean"]
```

---

### `RmseLoss_flow_temp` ← **Recommended for coupled hydro-temperature training**

Trains simultaneously on streamflow and stream temperature. The total loss is a weighted sum:

`L = w1 * L_flow + w2 * L_temp`

Where `L_flow` uses the same combined linear+log-sqrt RMSE as above, and `L_temp` is a simple RMSE on temperature. Both targets must be present in the observation file.

The weights `w1` and `w2` control the trade-off. Since flow and temperature have different scales and variability, you typically need to tune these. A starting point is `w1=3.0, w2=1.0` (flow 3× more important).

**When to use:** The main use case for this framework — running SNTEMP together with a hydrology model.

**Config:**
```yaml
loss_function: "RmseLoss_flow_temp"
loss_function_weights:
    w1: 3.0
    w2: 1.0
target: ["00060_Mean", "00010_Mean"]
temp_model_name: "SNTEMP"
```

---

### `NSEsqrtLoss_flow_temp`

A scale-normalized **Nash-Sutcliffe Efficiency (NSE)** loss applied to both streamflow and temperature simultaneously. Each term is computed on square-root-transformed values and normalized by its respective observed standard deviation across basins (the batch-normalized NSE formulation from Kratzert et al., 2019), so no manual weight tuning is required:

`L = L_NSE_flow + L_NSE_temp`

Unlike RMSE, this loss is scale-independent — basins with small flows and basins with large flows are weighted equally. This matters when training on basins whose flow magnitudes span several orders of magnitude.

**When to use:** Coupled hydro-temperature training on large, heterogeneous datasets (hundreds of basins).

**Config:**
```yaml
loss_function: "NSEsqrtLoss_flow_temp"
target: ["00060_Mean", "00010_Mean"]
temp_model_name: "SNTEMP"
```

---

## 8. Input Data Format

### Forcing Data (time-varying)

A NumPy array saved as `.npy` or a Feather dataframe (`.feather`), with shape:

```
[num_time_steps, num_basins, num_variables]
```

Each variable must be in the same order as listed in `varT_NN`, `varT_hydro_model`, and `varT_temp_model`. Variables listed in one section but not another are fine — the data loader selects columns by name.

| Variable | Description | Typical units |
|----------|-------------|---------------|
| `prcp(mm/day)` | Daily precipitation | mm/day |
| `tmean(C)` | Mean daily air temperature | °C |
| `tmin(C)` | Minimum daily air temperature | °C |
| `tmax(C)` | Maximum daily air temperature | °C |
| `PET_hargreaves(mm/day)` | Potential evapotranspiration (Hargreaves) | mm/day |
| `srad(W/m2)` | Incoming solar radiation | W/m² |
| `vp(Pa)` | Vapor pressure | Pa |
| `ccov` | Cloud cover fraction | 0–1 |

Missing values should be `NaN`. The data loader substitutes the mean for forcing inputs (to avoid model crashes) and masks NaN values in observation targets (so they do not contribute to the loss).

### Basin Attribute Data (static)

A NumPy array or Feather dataframe with shape:

```
[num_basins, num_attributes]
```

Each row corresponds to one basin in the same order as the forcing data. The variable names must match those listed in `varC_NN`, `varC_hydro_model`, and `varC_temp_model`. The config file contains three pre-built attribute lists for different common datasets (MSWEP/GAGES-II, CAMELS, Yalan's CAMELS) — uncomment the one matching your data.

### Observation Data (targets)

Same format as forcing data:

```
[num_time_steps, num_basins, num_target_variables]
```

The order of target variables must match the `target` list in the config. Days and basins with no observations should be `NaN` — these are masked automatically.

---

## 9. Running the Model

### 9.1 Hydrology Only (No Temperature)

To train a streamflow model without any temperature component:

**Step 1 — Edit `config/config_hydro_temp.yaml`:**
```yaml
# Choose a hydrology model
hydro_model_name: "marrmot_PRMS"  # or "HBV", "HBV_capillary"

# Disable temperature model
temp_model_name: "None"

# Use flow-only loss function
loss_function: "RmseLoss_flow_comb"

# Only streamflow as target
target: ["00060_Mean"]

# Set training period
t_train: [20000101, 20101231]
t_test:  [19900101, 19991231]
warm_up: 365

# Train mode
Action: [0]
EPOCHS: 100
saveEpoch: 10
```

**Step 2 — Run training:**
```bash
python main_hydro_temp.py
```

You will see output like:
```
1  from 42  in the  1 th epoch, and Loss is  0.523142
2  from 42  in the  1 th epoch, and Loss is  0.491832
...
Epoch 1 Loss 0.501237, time 48.32 sec, 3821 Kb allocated GPU memory
```

**Step 3 — Test with saved model:**

Edit the config:
```yaml
Action: [1]
EPOCH_testing: 100    # load the checkpoint from epoch 100
```

Run again:
```bash
python main_hydro_temp.py
```

Or train and test in one run:
```yaml
Action: [0, 1]
```

---

### 9.2 Integrated Hydrology + Temperature

To run the full coupled hydrology and stream temperature model:

**Edit `config/config_hydro_temp.yaml`:**
```yaml
# Hydrology model — produces the source flows for SNTEMP
hydro_model_name: "marrmot_PRMS"

# Enable stream temperature model
temp_model_name: "SNTEMP"
routing_temp_model: True

# Residence time filters (adjust for your basin characteristics)
res_time_lenF_srflow:      1
res_time_lenF_ssflow:      30
res_time_lenF_bas_shallow: 180
res_time_lenF_gwflow:      365

# Coupled loss function
loss_function: "RmseLoss_flow_temp"
loss_function_weights:
    w1: 3.0    # flow weight
    w2: 1.0    # temperature weight

# Both flow and temperature as targets
target: ["00060_Mean", "00010_Mean"]

# Add temperature forcing variables
varT_temp_model: ["ccov", "vp(Pa)", "srad(W/m2)"]

# Training settings
Action: [0, 1]
EPOCHS: 100
warm_up: 365
```

Run:
```bash
python main_hydro_temp.py
```

---

### 9.3 Temperature Only (with Pre-saved Flows)

If you have already trained a hydrology model and want to train or fine-tune the temperature model using pre-computed flows (instead of re-running hydrology every epoch):

**Edit `config/config_hydro_temp.yaml`:**
```yaml
# Skip the online hydrology model
hydro_model_name: "None"

# Point to pre-computed flow output from a previous run
flow_data_path: "D:\\models\\experiment_01\\out_dict_test.npy"

# Enable temperature model
temp_model_name: "SNTEMP"

# Coupled flow + temperature loss (both targets required)
loss_function: "RmseLoss_flow_temp"
loss_function_weights:
    w1: 3.0
    w2: 1.0
target: ["00060_Mean", "00010_Mean"]
```

Run:
```bash
python main_hydro_temp.py
```

This is faster because the hydrology model forward pass is skipped, and useful for exploring different temperature model settings without re-optimizing the hydrology parameters.

> The loss functions in this release all train on streamflow (and optionally temperature) together — there is no temperature-only loss. To emphasize temperature when the flows are already fixed, lower the flow weight `w1` relative to `w2`.

---

## 10. Understanding the Training Output

During training, each iteration prints:

```
5 from 42 in the 3 th epoch, and Loss is 0.312847
```
- `5` — current mini-batch iteration (out of 42 per epoch)
- `42` — total mini-batches per epoch (= num_basins / batch_size, rounded)
- `3` — current epoch number
- `0.312847` — loss value for this mini-batch (lower is better)

At the end of each epoch:
```
Epoch 3 Loss 0.298453, time 51.24 sec, 4102 Kb allocated GPU memory
```
- **Epoch loss** — average loss across all mini-batches. Should decrease over epochs.
- **Time** — wall-clock seconds for this epoch.
- **GPU memory** — kilobytes of GPU memory in use.

**What to watch for:**
- If the loss is not decreasing after several epochs, try increasing `EPOCHS`, or check that your data paths and variable names are correct.
- If you see `NaN` in gradient warnings, there may be numerical instability — check for missing data in your forcings.
- Epoch loss that oscillates wildly may indicate `batch_size` is too small. Try increasing it.

---

## 11. Output Files

After training, the output directory (set by `output_model`) contains a timestamped subdirectory:

```
output_model/
└── YYYYMMDD_HHMMSS/
    ├── model_Ep10.pt        # Saved PyTorch model — epoch 10
    ├── model_Ep20.pt        # Saved PyTorch model — epoch 20
    ├── ...
    ├── model_Ep100.pt       # Final model
    ├── out_dict_train.npy   # Training period predictions (dictionary saved as .npy)
    └── out_dict_test.npy    # Test period predictions
```

**Loading a saved model for inference:**
```python
import torch
model = torch.load("path/to/model_Ep100.pt")
model.eval()
```

**Reading the prediction dictionary:**
```python
import numpy as np
results = np.load("out_dict_test.npy", allow_pickle=True).item()

flow_sim = results["flow_sim"]   # shape: [time_steps, basins, 1]
temp_sim = results["temp_sim"]   # shape: [time_steps, basins, 1] (if temp model was used)
BFI_sim  = results["BFI_sim"]    # shape: [basins] — long-term baseflow index fraction
```

---

## 12. Tips and Common Issues

**Variable name mismatches:** The most common error is a variable listed in `varT_NN` or `varC_NN` that does not exist in your data file. The error message will tell you the name that was not found. Double-check spelling, capitalization, and units.

**Warm-up length:** If model states at the start of the test period look wrong (unrealistic initial snow or soil moisture), increase `warm_up` to 730 days (2 years). The first `warm_up` days are always discarded from the loss, so they are free in terms of training signal.

**GPU out of memory:** Reduce `batch_size` (fewer basins per iteration) or `rho` (shorter sequences). A good starting point for an 8 GB GPU is `batch_size=50, rho=365`. For a 24 GB GPU you can use `batch_size=100, rho=365`.

**Loss not decreasing:** Try `EPOCHS: 200` and check whether the loss eventually converges. If not, verify that your observation data is in the same units as the model output (streamflow in m³/s, temperature in °C). Also confirm that `t_train` and `tRange` are consistent with the date range in your forcing files.

**Resuming training from a checkpoint:** Load the checkpoint and continue training:
```python
import torch
model = torch.load("model_Ep50.pt")
optim = torch.optim.Adadelta(model.parameters())
# then pass model and optim back to train_differentiable_model()
```

**Reproducibility:** Always set `randomseed: [0]` (or any fixed integer) and keep `torch.backends.cudnn.deterministic = True` (already set in the training loop) to ensure identical results across runs.

---

## 13. Reproducing the Data-Release Figures

The `examples/` folder contains stand-alone scripts that regenerate the figures from the data release. Each script loads pre-computed model output and observation files, performs the necessary post-processing (trends, baseflow separation, statistics), and saves a publication-quality PNG.

| Script | Figure(s) it produces |
|--------|-----------------------|
| `examples/Fig1a.py` | Fig. 1a — observed vs. simulated mean-annual recharge (δ models vs. WaterGAP2-2c) |
| `examples/Fig1b.py` | Fig. 1b — wet/dry × warm/cold climate-region map of CONUS |
| `examples/Fig1cdef.py` | Fig. 1c–f — recharge vs. precipitation scatter by climate region |
| `examples/Fig2.py` | Fig. 2 — baseflow-trend scatter by zone (δ, GHM ensemble, DBH vs. obs) |
| `examples/Fig3_S3_S4.py` | Fig. 3 and Supp. Figs. S3, S4 — recharge mean/trend maps per period |
| `examples/Fig4_S2_S9.py` | Fig. 4 and Supp. Figs. S2, S9 — per-zone recharge time series |
| `examples/FigS1.py` | Supp. Fig. S1 — streamflow KGE + stream-temperature NSE/RMSE boxplots |
| `examples/FigS5_S6_S7_S8.py` | Supp. Figs. S5–S8 — per-zone boxplots of recharge trends |

### 13.1 What the Scripts Do

These scripts are **evaluation / plotting** scripts — they do **not** train or run the model. They consume:

- **Model output** the δPL framework already produced (streamflow, stream temperature, recharge), plus
- **Reference datasets** (USGS observations, ISIMIP2a/2b global hydrological model output, climate classifications, basin attributes, CONUS zone shapefiles).

All of these are distributed with the data release. You only need to point the scripts at the folder where you unpacked that data.

### 13.2 How to Run a Figure Script

**1. Install the environment.** Everything the figure scripts need (`cartopy`, `geopandas`, `xarray`, `netcdf4`, `baseflow`, `statsmodels`, `scikit-learn`, …) is already declared in `pyproject.toml`, so a single `uv sync` (see [Section 3](#3-environment-setup)) installs it all. No GPU is required for the figures.

**2. Tell the script where the data lives.** Open the script and edit the one line near the top:

```python
BASE_DIR = Path("D:/DR")        # ← change to wherever you unpacked the data release
# BASE_DIR = Path("/scratch/.../DR")   # example for Linux/HPC
```

Every other path in the script is derived from `BASE_DIR`, so this is the only edit you normally need.

**3. Run from the repository root.** The scripts import the local `post` package (`from post.read_GHMs_dPLs import ...`). That package lives at the repository root and is only importable when the root is on Python's path, so run the scripts **as modules from the repo root** — *not* by `cd`-ing into `examples/`:

```bash
# from the dPL_Hydro_SNTEMP_DR repository root, with the venv activated
python -m examples.Fig1a
python -m examples.Fig1b
python -m examples.Fig1cdef
python -m examples.Fig2
python -m examples.Fig3_S3_S4
python -m examples.Fig4_S2_S9
python -m examples.FigS1
python -m examples.FigS5_S6_S7_S8
```

> Note the **`-m examples.Fig1a`** form (dotted module name, no `.py`). Running `python examples/Fig1a.py` instead will fail with `ModuleNotFoundError: No module named 'post'`, because Python then puts `examples/` — not the repo root — on the import path. If you prefer the file form, add the repo root to `PYTHONPATH` first:
>
> ```powershell
> # Windows PowerShell
> $env:PYTHONPATH = (Get-Location)
> python examples\Fig1a.py
> ```
> ```bash
> # Linux / macOS
> PYTHONPATH=. python examples/Fig1a.py
> ```

Each script prints progress and finishes with `END`. The PNG(s) are written to `BASE_DIR/evaluation_figures/` (created automatically). The scripts use the non-interactive `Agg` Matplotlib backend, so they run fine over SSH or in a batch job with no display attached.

### 13.3 Setting Up the Data Directory

Unpack the data release so that `BASE_DIR` contains the `data/` and `M/` sub-trees below. `evaluation_figures/` is created for you. (Filenames are taken verbatim from the scripts — if you rename a file, update the corresponding constant at the top of the script.)

```
D:/DR/                                  ← BASE_DIR
├── data/                               # reference datasets & basin metadata
│   ├── ISIMIP2a/GHMs/Watergap2_2c/GSWP3/
│   │     watergap2-2c_gswp3_nobc_hist_nosoc_co2_qr_global_monthly_1901_2010.nc4
│   ├── ISIMIP2b/precip_dict.pkl
│   ├── recharge_208basins/huc12_recharge_208.csv
│   ├── ts_data_4231/
│   │     attr_HUC12_4231_grid_clip_20250224.npy
│   │     attr_HUC12_4231_grid_clip_20250224_name.json
│   ├── ts_2003basins/
│   │     attr2003_mswep_03122024.npy
│   │     attr2003_mswep_03122024_name.json
│   ├── tr_1223basins/attr1223_1023_daymet_20240826.npy
│   ├── climate_classifications/
│   │     climate_class_MIROC5_rcp60_2006_2099_USA.nc
│   │     2b_climate_labels_rcp60.json
│   │     2b_climate_labels_rcp85.json
│   └── Zones/Zones_0228.shp            # plus the .dbf / .shx / .prj sidecar files
│
├── M/                                  # model output (δPL) and cached intermediates
│   ├── daymet_1223_1023_PUB/                  # δ streamflow & temperature output
│   ├── daymet_1223_1023_PUB_huc12_dmt_4231/   # δ HUC12 recharge output
│   ├── upscale/
│   │     mSS_recharge_grid_huc12_dmt_4231_1980_2023.nc
│   │     mSS_recharge_grid_huc12_GSWP3_4238_1962_2011.nc
│   │     obs_mean_recharge_grid.nc
│   │     CPI/grid_gage_2003_CPI_intersect.feather
│   │     CPI/merged_shpfile_GHMs.shp
│   ├── qr_2b/sim_recharge_dict_yearly_2007_2100.npz
│   ├── bf/                                     # cached daily baseflow arrays (Fig2)
│   └── qtot/sim_qtot_dict.npz                  # streamflow cache (FigS1, auto-created)
│
└── evaluation_figures/                 # ← all PNG outputs land here (auto-created)
```

> A `.shp` shapefile is only valid alongside its sidecar files (`.dbf`, `.shx`, `.prj`, …). Keep the whole set together when copying `Zones_0228.shp` or `merged_shpfile_GHMs.shp`.

### 13.4 Script-by-Script Reference

Below are the specific inputs each script reads (all relative to `BASE_DIR`) and the file(s) it writes to `evaluation_figures/`.

**`Fig1a.py`** — recharge scatter, δ vs. WaterGAP2-2c vs. observations.
- Reads: `data/ISIMIP2a/GHMs/Watergap2_2c/GSWP3/…qr…1901_2010.nc4`, `data/recharge_208basins/huc12_recharge_208.csv`, `data/ts_data_4231/attr_HUC12_4231_grid_clip_20250224.npy(+_name.json)`, `M/daymet_1223_1023_PUB_huc12_dmt_4231/`, `M/upscale/mSS_recharge_grid_huc12_*.nc`, `M/upscale/obs_mean_recharge_grid.nc`.
- Writes: `rech_dPL_huc12_dmt_GSWP3_hist.png`.

**`Fig1b.py`** — CONUS climate-region map.
- Reads: `data/climate_classifications/climate_class_MIROC5_rcp60_2006_2099_USA.nc`.
- Writes: `climate_regions_classification_MIROC5_rcp60_2006_2099_USA.png`.
- **First run needs internet:** cartopy downloads the Natural Earth `admin_0_countries` (50 m) shapefile used to mask everything outside the USA.

**`Fig1cdef.py`** — recharge-vs-precipitation scatter by climate region (2 × 2).
- Reads: `M/qr_2b/sim_recharge_dict_yearly_2007_2100.npz`, `data/ISIMIP2b/precip_dict.pkl`, `data/climate_classifications/2b_climate_labels_rcp60.json` and `…rcp85.json`.
- Writes: `rech_precip_2008_2099.png`.

**`Fig2.py`** — per-zone baseflow-trend scatter (δ, GHM ensemble, DBH vs. obs).
- Controlled by the flag `LOAD_BF_FROM_DISK` (default **`True`**):
  - `True` → loads cached baseflow arrays from `M/bf/` (`mSS_Furey_daily.npy`, `mSS_Daymet_Furey_daily.npy`, `GHM_ens_Furey_daily.npy`, `DBH_Furey_daily.npy`, `obs_flow_Furey_daily.npy`). These are shipped with the release; this is the fast path.
  - `False` → re-runs the full baseflow-separation pipeline from raw ISIMIP2a GHM streamflow. This additionally requires the raw GHM NetCDFs (the reader defaults to `D:\P\inputs\isimip2a\GHMs`; edit `read_GHM_ISIMIP2a_daily` in `post/read_GHMs_dPLs.py` if your raw data is elsewhere) and writes the `.npy` caches into `M/bf/`.
- Also reads: `data/ts_2003basins/attr2003_mswep_03122024.npy(+_name.json)`, `data/tr_1223basins/attr1223_1023_daymet_20240826.npy`, `data/Zones/Zones_0228.shp`, `M/upscale/CPI/grid_gage_2003_CPI_intersect.feather`, `M/daymet_1223_1023_PUB/`.
- Writes: `bf_trend_hres_scatter_zones_dmt_GSWP3_zone0228_DBH.png`.

**`Fig3_S3_S4.py`** — recharge mean & trend maps, one figure per period (2008–2050, 2050–2099, 2008–2099).
- Reads: `M/qr_2b/sim_recharge_dict_yearly_2007_2100.npz`, `data/ts_2003basins/attr2003_mswep_03122024.npy(+_name.json)`, `data/Zones/Zones_0228.shp`.
- Writes: `map1_shp_recharge_dPL_GHM_2b_ens_rcp85_rcp60_{from}_{to}.png` (3 files).

**`Fig4_S2_S9.py`** — per-zone yearly recharge time series (3 figures: absolute, relative, and percent change).
- Reads: same `qr_2b` npz, `attr2003`, and `Zones` as above.
- Writes: `plot_future_trend_recharge_mvAve.png`, `plot_relative_future_trend_recharge_mvAve.png`, `perc_plot_relative_future_trend_recharge_mvAve.png`.

**`FigS1.py`** — streamflow KGE + stream-temperature NSE/RMSE boxplots.
- Streamflow is cached at `M/qtot/sim_qtot_dict.npz`: if present it is loaded; if absent it is **built from scratch** (reads raw ISIMIP2a GHM streamflow from `D:\P\inputs\isimip2a\GHMs` plus δ output from `M/daymet_1223_1023_PUB/`, then saves the cache). Stream temperature is always read from `M/daymet_1223_1023_PUB/`.
- Also reads: `data/ts_2003basins/attr2003_mswep_03122024.npy(+_name.json)`, `data/Zones/Zones_0228.shp`, `M/upscale/CPI/grid_gage_2003_CPI_intersect.feather`.
- Writes: `qtot_STemp.png`.

**`FigS5_S6_S7_S8.py`** — per-zone boxplots of recharge trends, one figure per (RCP, period) combination.
- Reads: `M/qr_2b/sim_recharge_dict_yearly_2007_2100.npz`, `attr2003`, `Zones`.
- Writes: `qr_trends_future_2b_zones_GHM_dPL_ens_{rcp}_{from}_{to}.png` (4 files).

### 13.5 Common Issues

**`ModuleNotFoundError: No module named 'post'`** — You ran the script from inside `examples/`. Run it from the repository root with the module form (`python -m examples.Fig1a`) or set `PYTHONPATH` to the repo root (see [Section 13.2](#132-how-to-run-a-figure-script)).

**`FileNotFoundError` on a `data/…` or `M/…` path** — `BASE_DIR` is not pointing at your unpacked data release, or the file was renamed. Fix the `BASE_DIR` line, or the specific path constant near the top of the script. `Fig2.py` raises a deliberate, descriptive `FileNotFoundError` listing exactly which `M/bf/*.npy` caches are missing when `LOAD_BF_FROM_DISK=True`.

**Shapefile read error** — A `.shp` was copied without its `.dbf` / `.shx` / `.prj` sidecars. Copy the entire shapefile set.

**`Fig1b.py` hangs or errors on first run** — It is downloading the Natural Earth boundaries. Ensure the machine has internet access for the first run (cartopy caches the file afterward).

**Recomputing instead of using caches is slow / needs raw data** — `Fig2.py` (`LOAD_BF_FROM_DISK=False`) and `FigS1.py` (missing `qtot` cache) fall back to reading raw ISIMIP2a GHM NetCDFs from `D:\P\inputs\isimip2a\GHMs`. If you only have the released caches, keep `LOAD_BF_FROM_DISK=True` and the `M/qtot/sim_qtot_dict.npz` cache in place so these raw reads are skipped.
