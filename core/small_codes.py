import torch
import numpy as np
import pandas as pd
import os
from core.read_configurations import config
import datetime as dt

def make_tensor(*values, has_grad=True, dtype=torch.float32, device=config['device']):

    if len(values) > 1:
        tensor_list = []
        for value in values:
            t = torch.tensor(value, requires_grad=has_grad, dtype=dtype, device=device)
            tensor_list.append(t)
    else:
        for value in values:
            tensor_list = torch.tensor(value, requires_grad=has_grad, dtype=dtype, device=device)
    return tensor_list

def create_output_dirs(args):
    if not os.path.exists(args['output']['model']):
        os.makedirs(args['output']['model'])
    out_folder = 'E_' + str(args['hyperparameters']['EPOCHS']) + \
                 '_R_' + str(args['hyperparameters']['rho']) + \
                 '_B_' + str(args['hyperparameters']['batch_size']) + \
                 '_H_' + str(args['hyperparameters']['hidden_size']) + \
                 '_dr_' + str(args['hyperparameters']['dropout'])
    if not os.path.exists(os.path.join(args['output']['model'], out_folder)):
        os.makedirs(os.path.join(args['output']['model'], out_folder))
    args['output']['out_dir'] = os.path.join(args['output']['model'], out_folder)
    return args


def t2dt(t, hr=False):
    tOut = None
    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception('hydroDL.utils.t2dt failed')
    return tOut


def tRange2Array(tRange, *, step=np.timedelta64(1, 'D')):
    sd = t2dt(tRange[0])
    ed = t2dt(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return C, ind1, ind2
