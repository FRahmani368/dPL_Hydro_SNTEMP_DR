import torch
import numpy as np
import pandas as pd
import os
from core.read_configurations import config

def make_tensor(value, has_garad=False, dtype=torch.float32, device=config['device']):
    t = torch.tensor(value, requires_grad=has_garad, dtype=dtype, device=device)
    return t

def create_output_dirs(args):
    if not os.path.exists(args['output']['model']):
        os.makedirs(args['output']['model'])
    out_folder = 'E_' + args['hyperparameters']['epochs'] + '_R_' + args['hyperparameters']['rho'] + \
                 '_B_' + args['hyperparameters']['batch_size'] +  '_H_' + args['hyperparameters']['hidden_size'] + \
                 '_dr_' + args['hyperparameters']['dropout']
    if not os.path.exists(os.path.join(args['output']['model'], out_folder)):
        os.makedirs(os.path.join(args['output']['model'], out_folder))
    args['output']['out_dir'] = os.path.join(args['output']['model'], out_folder)
    return args
