import sys

sys.path.append("../")
import torch
import os
from config.read_configurations import config_hydro_temp as config
from core.utils.randomseed_config import randomseed_config
from core.utils.small_codes import create_output_dirs
from MODELS.Differentiable_models import diff_hydro_temp_model
from MODELS.hydro_models.NWM_SACSMA.NWM_SACSMA_mul import NWM_SACSMA_Mul
from MODELS import train_test

def main():
    args = config
    # Define dummy data for x_hydro_model (time steps x grids x variables)
    time_steps, num_basins, num_vars = 730, 10, len(args["varT_hydro_model"])
    c_hydro_model = torch.zeros((num_basins,  len(args["varC_hydro_model"])), dtype=torch.float32).to(args["device"])
    x_hydro_model = torch.zeros((time_steps, num_basins,  len(args["varT_hydro_model"])),
                               dtype=torch.float32).to(args["device"])
    x_hydro_model[:,:,0] = x_hydro_model[:,:,0] + 6.0   # precip
    x_hydro_model[:, :, 1] = x_hydro_model[:, :, 1] + 25.0   # Tmean
    x_hydro_model[:, :, 2] = x_hydro_model[:, :, 2] + 2.0    # PET
    model = NWM_SACSMA_Mul().to(args["device"])
    # Define dummy parameters (num_basins x num_params)
    num_params = len(NWM_SACSMA_Mul().parameters_bound.keys())
    params_raw = 0.5 + torch.zeros((time_steps, num_basins, num_params, args["nmul"]), dtype=torch.float32).to(args["device"])
    # Define dummy conv_params_hydro (for routing)
    conv_params_hydro = torch.rand((num_basins, 2), dtype=torch.float32).to(args["device"])
    # Run forward pass
    output = model(
        x_hydro_model=x_hydro_model,
        c_hydro_model=c_hydro_model,
        params_raw=params_raw,
        args=args,
        routing=True,  # Set True if you want to include routing effects
        conv_params_hydro=conv_params_hydro
    )
    # Print results
    print("Model output keys:", output.keys())
    for key, value in output.items():
        print(f"{key}: {value.shape}")

if __name__ == "__main__":
    main()