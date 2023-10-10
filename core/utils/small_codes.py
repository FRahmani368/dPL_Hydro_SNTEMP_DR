import torch
import os
from core.load_data.time import tRange2Array
import json


def make_tensor(*values, has_grad=False, dtype=torch.float32, device="cuda"):

    if len(values) > 1:
        tensor_list = []
        for value in values:
            t = torch.tensor(value, requires_grad=has_grad, dtype=dtype, device=device)
            tensor_list.append(t)
    else:
        for value in values:
            if type(value) != torch.Tensor:
                tensor_list = torch.tensor(
                    value, requires_grad=has_grad, dtype=dtype, device=device
                )
            else:
                tensor_list = value.clone().detach()
    return tensor_list


def create_output_dirs(args):
    seed = args["randomseed"][0]
    # checking rho value first
    t = tRange2Array(args["t_train"])
    if t.shape[0] < args["rho"]:
        args["rho"] = t.shape[0]

    # checking the directory
    if not os.path.exists(args["output_model"]):
        os.makedirs(args["output_model"])
    out_folder = args["NN_model_name"] + \
            "_" + args["hydro_model_name"] + \
            "_" + args["temp_model_name"] + \
            '_E' + str(args['EPOCHS']) + \
             '_R' + str(args['rho']) + \
             '_B' + str(args['batch_size']) + \
             '_H' + str(args['hidden_size']) + \
             "_tr" + str(args["t_train"][0])[:4] + "_" + str(args["t_train"][1])[:4] + \
            "_ts" + str(args["t_test"][0])[:4] + "_" + str(args["t_test"][1])[:4] + \
            "_n" + str(args["nmul"]) + \
            "_" + str(seed)

    if not os.path.exists(os.path.join(args["output_model"], out_folder)):
        os.makedirs(os.path.join(args["output_model"], out_folder))
    # else:
    #     shutil.rmtree(os.path.join(args['output']['model'], out_folder))
    #     os.makedirs(os.path.join(args['output']['model'], out_folder))
    args["out_dir"] = os.path.join(args["output_model"], out_folder)

    # saving the args file in output directory
    config_file = json.dumps(args)
    config_path = os.path.join(args["out_dir"], "config_file.json")
    if os.path.exists(config_path):
        os.remove(config_path)
    f = open(config_path, "w")
    f.write(config_file)
    f.close()

    return args


def update_args(args, **kw):
    for key in kw:
        if key in args:
            try:
                args[key] = kw[key]
            except ValueError:
                print("Something went wrong in args when updating " + key)
        else:
            print("didn't find " + key + " in args")
    return args

