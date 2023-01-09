import numpy as np
import pandas as pd
import torch
import os
from core.read_configurations import config

def compute_infil(args, Net_rain, Net_ppt, Imperv_stor, Imperv_stor_max,
                  snowmelt, snowinfil_max, Net_snow, pkwater_equiv,
                  infil, HRU_type, Intcp_changeover):
    # compute runoff from cascading Hortonian flow  --> we don't have it yet, because there is only one HRU
    # if cascade ==0 --> avail_water = 0.0
    avail_water = torch.zeros(Net_rain.shape, dtype=torch.float32, device=args["device"])

    #compute runoff from canopy changeover water


def srunoff_smidx(args):
    # if Use_sroff_transfer==1 --> we don't have this yet. It is about reading transfer flow rate from external files

    #compute infiltration
    compute_infil(args, Net_rain, Net_ppt, Imperv_stor, Imperv_stor_max,
                  snowmelt, snowinfil_max, Net_snow, pkwater_equiv,
                  infil, HRU_type, Intcp_changeover)



def srunoff(args, **kwargs):
    if args["srunoff_module"] == "srunoff_smidx":
        retention_storage, evaporation, runoff_imperv = srunoff_smidx(args, kwargs["mean_air_temp"], kwargs["dayl"], kwargs["hamon_coef"])
    elif args["srunoff_module"] == "srunoff_carea":
        print("this surface runoff method is not ready yet")

    return retention_storage, evaporation, runoff_imperv