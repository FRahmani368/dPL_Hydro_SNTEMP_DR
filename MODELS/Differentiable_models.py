import torch.nn

from MODELS.hydro_models.marrmot_PRMS.prms_marrmot import prms_marrmot
from MODELS.hydro_models.marrmot_PRMS_mod.prms_marrmot_mod import prms_marrmot_mod
from MODELS.hydro_models.marrmot_PRMS_refreeze.prms_marrmot_refreeze import prms_marrmot_refreeze
from MODELS.hydro_models.marrmot_PRMS_gw0.prms_marrmot_gw0 import prms_marrmot_gw0
from MODELS.hydro_models.marrmot_PRMS_inteflow.prms_marrmot_interflow import prms_marrmot_interflow
from MODELS.hydro_models.HBV.HBVmul import HBVMul
from MODELS.hydro_models.SACSMA.SACSMAmul import SACSMAMul
from MODELS.hydro_models.SACSMA_with_snowpack.SACSMA_snow_mul import SACSMA_snow_Mul

from MODELS.temp_models.SNTEMP.SNTEMP import SNTEMP_flowSim
from MODELS.temp_models.SNTEMP_with_gw0.SNTEMP_gw0 import SNTEMP_flowSim_gw0

from MODELS.NN_models.LSTM_models import CudnnLstmModel
from MODELS.NN_models.MLP_models import MLPmul

from core.utils.small_codes import source_flow_calculation

# import MODELS
class diff_hydro_temp_model(torch.nn.Module):
    def __init__(self, args):
        super(diff_hydro_temp_model, self).__init__()
        self.args = args
        self.get_model()

    def get_NN_model_dim(self) -> None:
        self.nx = len(self.args["varT_NN"] + self.args["varC_NN"])

        # output size of NN
        if self.args["hydro_model_name"] != "None":
            if self.args["routing_hydro_model"] == True:  # needs a and b for routing with conv method
                self.ny_hydro = self.args["nmul"] * (len(self.hydro_model.parameters_bound)) + len(
                    self.hydro_model.conv_routing_hydro_model_bound)
            else:
                self.ny_hydro = self.args["nmul"] * len(self.hydro_model.parameters_bound)
        else:
            self.ny_hydro = 0

        # SNTEMP  # needs a and b for calculating different source flow temperatures with conv method
        if self.args["temp_model_name"] != "None":
            self.ny_temp = self.args["nmul"] * len(self.temp_model.parameters_bound)
            if self.args["lat_temp_adj"] == True:
                self.ny_temp = self.ny_temp + self.args["nmul"]
            if self.args["routing_temp_model"] == True:
                self.ny_temp = self.ny_temp + len(self.temp_model.conv_temp_model_bound)
        else:
            self.ny_temp = 0
        # if self.args["hydro_model_name"] == "HBV":   # no need to have a PET to AET coef
        #     self.ny_PET = 0
        # elif self.args["hydro_model_name"] == "marrmot_PRMS":   # need a PET to AET coef
        #     self.ny_PET = self.args["nmul"]
        # if self.args["potet_module"] in ["potet_hargreaves", "potet_hamon", "dataset"]:
        #     self.ny_PET = self.args["nmul"]
        self.ny = self.ny_hydro + self.ny_temp # + self.ny_PET

    def get_model(self) -> None:
        # hydro_model_initialization
        if self.args["hydro_model_name"] != "None":
            if self.args["hydro_model_name"] == "marrmot_PRMS":
                self.hydro_model = prms_marrmot()
            elif self.args["hydro_model_name"] == "marrmot_PRMS_mod":
                self.hydro_model = prms_marrmot_mod()
            elif self.args["hydro_model_name"] == "marrmot_PRMS_refreeze":
                self.hydro_model = prms_marrmot_refreeze()
            elif self.args["hydro_model_name"] == "marrmot_PRMS_gw0":
                self.hydro_model = prms_marrmot_gw0()
            elif self.args["hydro_model_name"] == "marrmot_PRMS_interflow":
                self.hydro_model = prms_marrmot_interflow()
            elif self.args["hydro_model_name"] == "HBV":
                self.hydro_model = HBVMul(self.args)
            elif self.args["hydro_model_name"] == "SACSMA":
                self.hydro_model = SACSMAMul()
            elif self.args["hydro_model_name"] == "SACSMA_with_snow":
                self.hydro_model = SACSMA_snow_Mul()
            else:
                print("hydrology (streamflow) model type has not been defined")
                exit()
            # temp_model_initialization
        if self.args["temp_model_name"] != "None":
            if self.args["temp_model_name"] == "SNTEMP":
                self.temp_model = SNTEMP_flowSim()  # this model needs a hydrology model as backbone
            elif self.args["temp_model_name"] == "SNTEMP_gw0":
                self.temp_model = SNTEMP_flowSim_gw0()  # this model needs a hydrology model as backbone, and 4 outflow
            else:
                print("temp model type has not been defined")
                exit()
        # get the dimensions of NN model based on hydro modela and temp model
        self.get_NN_model_dim()
        # NN_model_initialization
        if self.args["NN_model_name"] == "LSTM":
            self.NN_model = CudnnLstmModel(nx=self.nx,
                                           ny=self.ny,
                                           hiddenSize=self.args["hidden_size"],
                                           dr=self.args["dropout"])
        elif self.args["NN_model_name"] == "MLP":
            self.NN_model = MLPmul(self.args, nx=self.nx, ny=self.ny)
        else:
            print("NN model type has not been defined")
            exit()

    def breakdown_params(self, params_all):
        params_dict = dict()
        params_hydro_model = params_all[:, :, :self.ny_hydro]
        params_temp_model = params_all[:, :, self.ny_hydro: (self.ny_hydro + self.ny_temp)]
        # if self.ny_PET > 0:
        #     params_dict["params_PET_model"] = torch.sigmoid(params_all[-1, :, (self.ny_hydro + self.ny_temp):])
        # else:
        #     params_dict["params_PET_model"] = None


        # Todo: I should separate PET model output from hydro_model and temp_model.
        #  For now, evap is calculated in both models individually (with same method)

        if self.args['hydro_model_name'] != "None":
            # hydro params
            params_dict["hydro_params_raw"] = torch.sigmoid(
                params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.args["nmul"]]).view(
                params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
                self.args["nmul"])
            # routing params
            if self.args["routing_hydro_model"] == True:
                params_dict["conv_params_hydro"] = torch.sigmoid(
                    params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.args["nmul"]:])
            else:
                params_dict["conv_params_hydro"] = None

        if self.args['temp_model_name'] != "None":
            # if lat_temp_adj is True --> dim:[days, basins, 5,  nmul]   , because there are 5 params in temp model
            if self.args["lat_temp_adj"] == True:
                params_dict["temp_params_raw"] = torch.sigmoid(
                    params_temp_model[:, :,
                    :(len(self.temp_model.parameters_bound) + len(self.temp_model.lat_adj_params_bound)) * self.args[
                        "nmul"]]).view(
                    params_temp_model.shape[0], params_temp_model.shape[1],
                    len(self.temp_model.parameters_bound) + len(self.temp_model.lat_adj_params_bound),
                    self.args["nmul"])
            # if lat_temp_adj is False --> dim:[days, basins, 5 + 1,  nmul]
            else:
                params_dict["temp_params_raw"] = torch.sigmoid(
                    params_temp_model[:, :, :len(self.temp_model.parameters_bound) * self.args["nmul"]]).view(
                    params_temp_model.shape[0], params_temp_model.shape[1], len(self.temp_model.parameters_bound),
                    self.args["nmul"])
            # convolution parameters for ss and gw temp calculation
            if self.args["routing_temp_model"] == True:
                params_dict["conv_params_temp"] = torch.sigmoid(params_temp_model[-1, :, -len(self.temp_model.conv_temp_model_bound):])
            else:
                print("it has not been defined yet what approach should be taken in place of conv")
                exit()
        return params_dict


    def forward(self, dataset_dictionary_sample):
        params_all = self.NN_model(dataset_dictionary_sample["inputs_NN_scaled"])   # [self.args["warm_up"]:, :, :]
        # breaking down the parameters to different pieces for different models (PET, hydro, temp)
        params_dict = self.breakdown_params(params_all)
        if self.args['hydro_model_name'] != "None":
            # hydro model
            flow_out = self.hydro_model(
                dataset_dictionary_sample["x_hydro_model"],
                dataset_dictionary_sample["c_hydro_model"],
                params_dict['hydro_params_raw'],
                self.args,
                # PET_param=params_dict["params_PET_model"],  # PET is in both temp and flow model
                warm_up=self.args["warm_up"],
                routing=self.args["routing_hydro_model"],
                conv_params_hydro=params_dict["conv_params_hydro"]
            )
            # baseflow index percentage
            ## means we are using two deep groundwater buckets named gwflow & bas_shallow
            if "bas_shallow" in flow_out.keys():
                baseflow = flow_out["gwflow"] + flow_out["bas_shallow"]
            else:
                baseflow = flow_out["gwflow"]
            flow_out["BFI_sim"] = 100 * (torch.sum(baseflow, dim=0) / (
                    torch.sum(flow_out["flow_sim"], dim=0) + 0.00001))[:, 0]

            if self.args['temp_model_name'] != "None":
                # source flow calculation and converting mm/day to m3/ day
                source_flows_dict = source_flow_calculation(self.args, flow_out,
                                                            dataset_dictionary_sample[
                                                                "c_NN"],
                                                            after_routing=True)
                # temperature model
                temp_out = self.temp_model.forward(dataset_dictionary_sample["x_temp_model"],
                                                   dataset_dictionary_sample["airT_mem_temp_model"],
                                                   dataset_dictionary_sample["c_temp_model"],
                                                   params_dict["temp_params_raw"],
                                                   conv_params_temp=params_dict["conv_params_temp"],
                                                   args=self.args,
                                                   PET=flow_out["PET_hydro"] * (1 / (1000 * 86400)),
                                                   # converting mm/day to m/sec,
                                                   source_flows=source_flows_dict)

                return {**flow_out, **temp_out}  # combining both dictionaries
            else:
                return flow_out
        else:
            ## flow data comes from a dataset (pre-saved model's output, or observations)
            if self.args['temp_model_name'] != "None":
                # source flow calculation and converting mm/day to m3/ day
                source_flows_dict = source_flow_calculation(self.args, dataset_dictionary_sample,
                                                            dataset_dictionary_sample[
                                                                "c_NN"],
                                                            after_routing=True)
                # temperature model
                temp_out = self.temp_model.forward(dataset_dictionary_sample["x_temp_model"],
                                                   dataset_dictionary_sample["airT_mem_temp_model"],
                                                   dataset_dictionary_sample["c_temp_model"],
                                                   params_dict["temp_params_raw"],
                                                   conv_params_temp=params_dict["conv_params_temp"],
                                                   args=self.args,
                                                   PET=dataset_dictionary_sample["PET_hydro"] * (
                                                               1 / (1000 * 86400)),
                                                   # converting mm/day to m/sec,
                                                   source_flows=source_flows_dict)

                return temp_out  # combining both dictionaries
