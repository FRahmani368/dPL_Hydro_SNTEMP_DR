from MODELS.loss_functions import (RmseLoss_flow_comb,
                                   RmseLoss_flow_temp,
                                   RmseLoss_flow_temp_BFI_PET)
import importlib
def get_lossFun(args):
    # module = importlib.import_module(args["loss_function"])
    spec = importlib.util.spec_from_file_location(args["loss_function"], "./MODELS/loss_functions/" + args["loss_function"] + ".py")
    module = spec.loader.load_module()
    loss_function_default = getattr(module, args["loss_function"])
    if (args["loss_function"] =="RmseLoss_flow_temp") or (args["loss_function"] =="RmseLoss_flow_temp_BFI") or \
            (args["loss_function"] =="RmseLoss_flow_temp_BFI_PET") or (args["loss_function"] =="RmseLoss_BFI_temp"):
        lossFun = loss_function_default(w1=args["loss_function_weights"]["w1"],
                                        w2=args["loss_function_weights"]["w2"])
    else:
        lossFun = loss_function_default()
    return lossFun

