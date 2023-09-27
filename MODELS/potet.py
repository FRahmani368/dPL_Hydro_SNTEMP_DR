import numpy as np
import pandas as pd
import torch
import os
# from core.read_configurations import config


def potet_hamon(mean_air_temp, dayl, hamon_coef=0.0055):  # hamon_coef=0.1651
    """
    :param mean_air_temp: daily mean air temperature (celecius)
    :param dayl: seconds of sunshine(number of hours between sunshine and sunset), need to convert to hour
    :param hamon_coef: coefficient for Hamon equation
    :return: PET potential evapotranspiration (m/sec after multiplying to conversion factors)
    """

    e_s = 6.108 * torch.exp(17.26939 * mean_air_temp / (mean_air_temp + 237.3))  # mbar

    # rho is saturated water-vapor density (absolute humidity)
    rho = (216.7 / (mean_air_temp + 273.3)) * e_s

    PET = (
        hamon_coef * torch.pow((dayl / 3600) / 12, 2) * rho * 0.0254 / 86400
    )  # 25.4 is converting inches/day to m/s

    # replacing negative values with zero
    mask_PET = PET.ge(0)
    PET = PET * mask_PET.int().float()

    return PET

def potet_hargreaves(tmin, tmax, tmean, lat, day_of_year):
    # calculate the day of year
    # dfdate = pd.date_range(start=str(trange[0]), end=str(trange[1]), freq='D', closed='left') # end not included
    # tempday = np.array(dfdate.dayofyear)
    # day_of_year = np.tile(tempday.reshape(-1, 1), [1, tmin.shape[-1]])
    # Loop to reduce memory usage
    pet = np.zeros(tmin.shape, dtype=np.float32) * np.NaN
    for ii in np.arange(len(pet[:, 0])):
        trange = tmax[ii, :] - tmin[ii, :]
        trange[trange < 0] = 0
        latitude = np.deg2rad(lat[ii, :])
        SOLAR_CONSTANT = 0.0820
        sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * day_of_year[ii, :] - 1.39))
        sha = np.arccos(np.clip(-np.tan(latitude) * np.tan(sol_dec), -1, 1))
        ird = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year[ii, :]))
        tmp1 = (24.0 * 60.0) / np.pi
        tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
        tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
        et_rad = tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)
        pet[ii, :] = 0.0023 * (tmean[ii, :] + 17.8) * trange ** 0.5 * 0.408 * et_rad
    pet[pet < 0] = 0
    return pet


def get_potet(args, **kwargs):
    if args["potet_module"] == "potet_hamon":
        PET = potet_hamon(kwargs["mean_air_temp"], kwargs["dayl"], kwargs["hamon_coef"])
    elif args["potet_module"] == "potet_pm":
        print("this PET method is not ready yet")
    elif args["potet_module"] == "potet_hargreaves":
        PET = potet_hamon(kwargs["tmin"], kwargs["tmax"], kwargs["tmean"], kwargs["lat"], kwargs["trange"])
    return PET
