import torch
import numpy as np
import pandas as pd
import os
from core.read_configurations import config
import datetime as dt
from core import hydroDL
from ruamel.yaml import YAML
import json
import shutil


def make_tensor(*values, has_grad=False, dtype=torch.float32, device=config["device"]):

    if len(values) > 1:
        tensor_list = []
        for value in values:
            t = torch.tensor(value, requires_grad=has_grad, dtype=dtype, device=device)
            tensor_list.append(t)
    else:
        for value in values:
            tensor_list = torch.tensor(
                value, requires_grad=has_grad, dtype=dtype, device=device
            )
    return tensor_list


def create_output_dirs(args, seed):
    # checking rho value first
    t = hydroDL.utils.time.tRange2Array(args["optData"]["t_train"])
    if t.shape[0] < args["hyperparameters"]["rho"]:
        args["hyperparameters"]["rho"] = t.shape[0]

    # checking the directory
    if not os.path.exists(args["output"]["model"]):
        os.makedirs(args["output"]["model"])
    # out_folder = 'E_' + str(args['hyperparameters']['EPOCHS']) + \
    #              '_R_' + str(args['hyperparameters']['rho']) + \
    #              '_B_' + str(args['hyperparameters']['batch_size']) + \
    #              '_H_' + str(args['hyperparameters']['hidden_size']) + \
    #              '_dr_' + str(args['hyperparameters']['dropout']) + "_" + str(seed)
    L = len(args["static_params_list"])
    if L > 0:
        stat = str(args["static_params_list"][0])
        if L > 1:
            for i in range(1, L):
                stat = stat + "_" + str(args["static_params_list"][i])
    else:
        stat = ""

    L = len(args["semi_static_params_list"])
    if L > 0:
        semi = str(args["semi_static_params_list"][0])
        if L > 1:
            for i in range(1, L):
                semi = semi + "_" + str(args["semi_static_params_list"][i])
    else:
        semi = ""

    out_folder = (
        str(args["res_time_params"]["type"])
        + "_gw_"
        + str(args["res_time_params"]["lenF_gwflow"])
        + "_ss_"
        + str(args["res_time_params"]["lenF_ssflow"])
        + "_adj_"
        + str(args["lat_temp_adj"][0])
        + "_fr_"
        + str(args["frac_smoothening"]["mode"][0])
        + str(args["frac_smoothening"]["gw_filter_size"])
        + "_stat_"
        + stat
        + "_semi_"
        + semi
        + "_nmul_"
        + str(args["nmul"])
        + "_s_"
        + str(seed)
    )

    # '_sh_' + str(args['shade_smoothening'][0]) +
    if not os.path.exists(os.path.join(args["output"]["model"], out_folder)):
        os.makedirs(os.path.join(args["output"]["model"], out_folder))
    # else:
    #     shutil.rmtree(os.path.join(args['output']['model'], out_folder))
    #     os.makedirs(os.path.join(args['output']['model'], out_folder))
    args["output"]["out_dir"] = os.path.join(args["output"]["model"], out_folder)

    # saving the args file in output directory
    config_file = json.dumps(args)
    config_path = os.path.join(args["output"]["out_dir"], "config_file.json")
    if os.path.exists(config_path):
        os.remove(config_path)
    f = open(config_path, "w")
    f.write(config_file)
    f.close()

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
        raise Exception("hydroDL.utils.t2dt failed")
    return tOut


def tRange2Array(tRange, *, step=np.timedelta64(1, "D")):
    sd = t2dt(tRange[0])
    ed = t2dt(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return C, ind1, ind2
