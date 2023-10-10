from MODELS.loss_functions import crit
def get_lossFun(args):
    if args["loss_function"] == "RmseLoss_temp_flow_BFI_PET":
        return crit.RmseLoss_temp_flow_BFI_PET()
    elif args["loss_function"] == "RmseLoss_temp_flow_BFI":
        return crit.RmseLoss_temp_flow_BFI
    elif args["loss_function"] == "RmseLoss_temp_flow":
        return crit.RmseLoss_temp_flow
