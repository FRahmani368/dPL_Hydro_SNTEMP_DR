import numpy as np
import pandas as pd
import torch
import os

class NRBacksolveFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        y0,
        theta,
        delta_t,
        input,
        nflux
    ):

        bs,ny = y0.shape
        rho = input.shape[0]

class prms_marrmot_2D(torch.nn.Module):
    def __init__(self, delta_t,climate_data):
        super().__init__()

        self.delta_t = delta_t
        self.climate_data = climate_data

    def forward(self, t, y, theta):
