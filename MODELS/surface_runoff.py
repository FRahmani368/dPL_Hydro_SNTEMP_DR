import numpy as np
import pandas as pd
import torch
import os
from core.read_configurations import config


# def perv_comp(args, Pptp, Ptc, Infil, Srp, Soil_moist, Smidx_coef, Smidx_exp, Carea_max):
def perv_comp(args, Pptp, Ptc, Infil, Srp, Carea_max, **kwargs):
    # this function is for pervious area computation
    # for srunoff_smidx, we need these inputs as kwargs: Soil_moist, Smidx_coef, Smidx_exp
    # for srunoff_carea, we need Carea_min, Carea_dif, Soil_rechr, Soil_rechr_max
    if args["srunoff_module"] == "srunoff_smidx":
        smidx = Soil_moist + 0.5 * Ptc
        ca_fraction = Smidx_coef * 10 ** (Smidx_exp * smidx)
    elif args["srunoff_module"] == "srunoff_carea":
        ca_fraction = Carea_min + Carea_dif * Soil_rechr / Soil_rechr_max

    # ca_fraction cannot be more than carea_max, no matter which module we are using
    ca_fraction = torch.where(ca_fraction > Carea_max, Carea_max, ca_fraction)
    srpp = ca_fraction * Pptp
    Contrib_fraction = ca_fraction   # not sure why we need this, it is in fortran code srunoff.f90 at line 929. probably we don'tneed it as it is not the output of the subroutine
    Infil = Infil - srpp
    Srp = Srp + srpp

    return Infil, Srp


def compute_infil(
    args,
    Net_rain,
    Net_ppt,
    Imperv_stor,
    Imperv_stor_max,
    snowmelt,
    snowinfil_max,
    Net_snow,
    pkwater_equiv,
    infil,
    HRU_type,
    Intcp_changeover,
    Srp,
        **kwargs
    # Soil_moist,
    # Smidx_coef,
    # Smidx_exp,
):
    # compute runoff from cascading Hortonian flow  --> we don't have it yet, because there is only one HRU
    # if cascade ==0 --> avail_water = 0.0
    avail_water = torch.zeros(
        Net_rain.shape, dtype=torch.float32, device=args["device"]
    )

    # compute runoff from canopy changeover water
    avail_water = avail_water + Intcp_changeover
    infil = infil + Intcp_changeover

    # pervious area computation
    # this function is for pervious area computation
    # for srunoff_smidx, we need these inputs as kwargs: Soil_moist, Smidx_coef, Smidx_exp
    # for srunoff_carea, we need Carea_min, Carea_dif, Soil_rechr, Soil_rechr_max
    if args["srunoff_module"] == "srunoff_smidx":
        infil_temp, Srp_temp = perv_comp(args,
                               avail_water, avail_water, infil, Srp, kwargs["Soil_moist"], kwargs["Smidx_coef"],
                               kwargs["Smidx_exp"]
                               )
    elif args["srunoff_module"] == "srunoff_carea":
    infil_temp, Srp_temp = perv_comp(args,
        avail_water, avail_water, infil, Srp, kwargs["Carea_min"], kwargs["Carea_dif"], kwargs["Soil_rechr"], kwargs["Soil_rechr_max"]
    )

    # perv_comp should be activated only for hru_type==1, then:
    infil = torch.where(HRU_type == 1, infil_temp, infil)
    Srp = torch.where(HRU_type == 1, Srp_temp, Srp)

    # pptmix_nopack ==1 means: net_rain > 0 & net_snow > o & pkwater_equiv ==0
    # pptmix_nopack is a Flag indicating that a mixed precipitation event has occurred
    # with no snowpack present on an HRU (1), otherwise (0)
    # if rain/snow event with no antecedent snowpack, compute the runoff from the rain first and then proceed with the
    # snowmelt computations, lines 839 - 847 in srunoff.f90
    avail_water = torch.where((Net_rain > 0) & (Net_snow > 0) & (pkwater_equiv > 0),
                              avail_water + Net_rain,
                              avail_water)
    infil = torch.where((Net_rain > 0) & (Net_snow > 0) & (pkwater_equiv > 0),
                              infil + Net_rain,
                              infil)
    if args["srunoff_module"] == "srunoff_smidx":
        infil_temp, Srp_temp = perv_comp(args,
                               Net_rain, Net_rain, infil, Srp, kwargs["Soil_moist"], kwargs["Smidx_coef"],
                               kwargs["Smidx_exp"]
                               )
    else:
        print("srunoff_carea is not ready yet")
    infil = torch.where(HRU_type == 1, infil_temp, infil)
    Srp = torch.where(HRU_type == 1, Srp_temp, Srp)



    # If precipitation on snowpack, all water available to the surface is
    # considered to be snowmelt, and the snowmelt infiltration
    # procedure is used.  If there is no snowpack and no precip,
    # then check for melt from last of snowpack.  If rain/snow mix
    # with no antecedent snowpack, compute snowmelt portion of runoff
    avail_water = torch.where(snowmelt > 0.0,
                              avail_water + snowmelt,
                              avail_water)
    infil = torch.where(snowmelt > 0.0,
                        infil + snowmelt,
                        infil)




    return infil, Srp

def srunoff_carea(
    args,
    Net_rain,
    Net_ppt,
    Imperv_stor,
    Imperv_stor_max,
    snowmelt,
    snowinfil_max,
    Net_snow,
    pkwater_equiv,
    infil,
    HRU_type,
    Intcp_changeover,
    Srp, Carea_min, Carea_dif, Soil_rechr, Soil_rechr_max
):
    infil, Srp = compute_infil(
        args,
        Net_rain,
        Net_ppt,
        Imperv_stor,
        Imperv_stor_max,
        snowmelt,
        snowinfil_max,
        Net_snow,
        pkwater_equiv,
        infil,
        HRU_type,
        Intcp_changeover,
        Srp, Carea_min, Carea_dif, Soil_rechr, Soil_rechr_max
    )

    return retention_storage, evaporation, runoff_imperv, infil, Srp



def srunoff_smidx(
    args,
    Net_rain,
    Net_ppt,
    Imperv_stor,
    Imperv_stor_max,
    snowmelt,
    snowinfil_max,
    Net_snow,
    pkwater_equiv,
    infil,
    HRU_type,
    Intcp_changeover,
    Srp, Soil_moist, Smidx_coef, Smidx_exp
):
    # if Use_sroff_transfer==1 --> we don't have this yet. It is about reading transfer flow rate from external files

    # compute infiltration
    infil, Srp = compute_infil(
        args,
        Net_rain,
        Net_ppt,
        Imperv_stor,
        Imperv_stor_max,
        snowmelt,
        snowinfil_max,
        Net_snow,
        pkwater_equiv,
        infil,
        HRU_type,
        Intcp_changeover,
        Srp, Soil_moist, Smidx_coef, Smidx_exp
    )

    return retention_storage, evaporation, runoff_imperv, infil, Srp


def srunoff(args, **kwargs):

    # sroff_flag==1 in fortran code
    if args["srunoff_module"] == "srunoff_smidx":
        retention_storage, evaporation, runoff_imperv, infil, Srp = srunoff_smidx(
            args,
            kwargs["Net_rain"],
            kwargs["Net_ppt"],
            kwargs["Imperv_stor"],
            kwargs["Imperv_stor_max"],
            kwargs["snowmelt"],
            kwargs["snowinfil_max"],
            kwargs["Net_snow"],
            kwargs["pkwater_equiv"],
            kwargs["infil"],
            kwargs["HRU_type"],
            kwargs["Intcp_changeover"],
            kwargs["Srp"],
            kwargs["Soil_moist"],
            kwargs["Smidx_coef"],
            kwargs["Smidx_exp"]
        )

    # sroff_flag=2 in fortran code
    elif args["srunoff_module"] == "srunoff_carea":
        print("this surface runoff method is not ready yet")
        retention_storage, evaporation, runoff_imperv, infil, Srp = srunoff_carea(
            args,
            kwargs["Net_rain"],
            kwargs["Net_ppt"],
            kwargs["Imperv_stor"],
            kwargs["Imperv_stor_max"],
            kwargs["snowmelt"],
            kwargs["snowinfil_max"],
            kwargs["Net_snow"],
            kwargs["pkwater_equiv"],
            kwargs["infil"],
            kwargs["HRU_type"],
            kwargs["Intcp_changeover"],
            kwargs["Srp"],
            kwargs["Soil_moist"],
            kwargs["Smidx_coef"],
            kwargs["Smidx_exp"]
        )
    return retention_storage, evaporation, runoff_imperv, infil, Srp
