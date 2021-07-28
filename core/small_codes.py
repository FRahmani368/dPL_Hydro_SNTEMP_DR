import torch
import numpy as np
import pandas as pd
import os
from core.read_configurations import config

def make_tensor(value, has_garad=False, dtype=torch.float32, device=config['device']):
    t = torch.tensor(value, requires_grad=has_garad, dtype=dtype, device=device)
    return t