
import sys
sys.path.append("../")
import torch
import os
from config.read_configurations import config_PRMS_SNTEMP as config
from core.utils.randomseed_config import randomseed_config
from core.utils.small_codes import create_output_dirs
from core.load_data.normalizing import init_norm_stats
from MODELS.loss_functions.crit import *
from MODELS.Differentiable_models import diff_hydro_temp_model
from MODELS import train_test



def main_marrmotPRMS_SNTEMP(args):
    # updating args. all settings are here
    # args = update_args(args,
    #                     res_time_type=typ,
    #                     res_time_lenF_gwflow=LenF_gw,
    #                     res_time_lenF_ssflow=LenF_ss,
    #                     lat_temp_adj=adj,
    #                     frac_smoothening_mode=frac_smooth,
    #                     randomseed=seed
    # )
    randomseed_config(seed=args["randomseed"][0])
    # Creating output directories and adding it to args
    args = create_output_dirs(args)
    # creating the stats for normalization
    init_norm_stats(args)

    # training
    if 0 in args["Action"]:
        diff_model = diff_hydro_temp_model(args)
        lossFun = globals()[args["loss_function"]]()
        optim = torch.optim.Adadelta(diff_model.parameters())
        train_test.train_differentiable_model(
            args=args,
            diff_model=diff_model,
            lossFun=lossFun,
            optim=optim
        )
    if 1 in args["Action"]:
        modelFile = os.path.join(args["out_dir"], "model_Ep" + str(args["EPOCHS"]) + ".pt")
        diff_model = torch.load(modelFile)
        train_test.test_differentiable_model(
            args=args,
            diff_model=diff_model
        )


if __name__ == "__main__":
    args = config
    main_marrmotPRMS_SNTEMP(args)
    print("END")
