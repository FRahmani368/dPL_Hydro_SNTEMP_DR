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


def get_potet(args, **kwargs):
    if args["potet_module"] == "potet_hamon":
        PET = potet_hamon(kwargs["mean_air_temp"], kwargs["dayl"], kwargs["hamon_coef"])
    elif args["potet_module"] == "potet_pm":
        print("this PET method is not ready yet")

    return PET
