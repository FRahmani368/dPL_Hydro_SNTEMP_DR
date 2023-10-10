import torch.nn

from MODELS.hydro_models.marrmot.prms_marrmot import prms_marrmot
from MODELS.hydro_models.HBV import HBVmul

from MODELS.temp_models.PGML_STemp import SNTEMP_flowSim

from MODELS.NN_models.LSTM_models import CudnnLstmModel
from MODELS.NN_models.MLP_models import MLPmul
# import MODELS
class diff_hydro_temp_model(torch.nn.Module):
    def __init__(self, args):
        super(diff_hydro_temp_model, self).__init__()
        self.args = args
        self.get_model()
    def get_model(self) -> None:
        # hydro_model_initialization
        if self.args["hydro_model_name"] == "marrmot_PRMS":
            self.hydro_model = prms_marrmot()
        elif self.args["hydro_model_name"] == "HBV":
            self.hydro_model = HBVmul()
        else:
            print("hydrology (streamflow) model type has not been defined")
            exit()
            # temp_model_initialization
        if self.args["temp_model_name"] == "SNTEMP":
            self.temp_model = SNTEMP_flowSim()  # this model needs a hydrology model as backbone
        else:
            print("temp model type has not been defined")
            exit()
        # NN_model_initialization
        self.nx, self.ny, self.ny_hydro, self.ny_temp, self.ny_PET = self.get_NN_model_dim()
        if self.args["NN_model_name"] == "LSTM":
            self.NN_model = CudnnLstmModel(nx=self.nx,
                                            ny=self.ny,
                                            hiddenSize=self.args["hidden_size"],
                                            dr=self.args["dropout"])
        elif self.args["NN_model_name"] == "MLP":
            self.NN_model = MLPmul(self.args)


    def get_NN_model_dim(self):
        nx = len(self.args["varT_NN"] + self.args["varC_NN"])

        # output size of NN
        if self.args["routing_hydro_model"] == True:  # needs a and b for routing with conv method
            ny_hydro = self.args["nmul"] * (len(self.hydro_model.parameters_bound)) + len(
                self.hydro_model.conv_routing_hydro_model_bound)
        else:
            ny_hydro = self.args["nmul"] * len(self.hydro_model.parameters_bound)

        # SNTEMP  # needs a and b for calculating different source flow temperatures with conv method
        if self.args["routing_temp_model"] == True:
            ny_temp = self.args["nmul"] * (len(self.temp_model.parameters_bound)) + len(
                self.temp_model.conv_temp_model_bound)
        else:
            ny_temp = self.args["nmul"] * len(self.temp_model.parameters_bound)
        if self.args["lat_temp_adj"] == True:
            ny_temp = ny_temp + self.args["nmul"]
        if self.args["potet_module"] in ["potet_hargreaves", "potet_hamon", "dataset"]:
            ny_PET = self.args["nmul"]
        ny = ny_hydro + ny_temp + ny_PET
        return nx, ny, ny_hydro, ny_temp, ny_PET

    def forward(self, inputs_NN, x_hydro_model, c_hydro_model, x_temp_model, c_temp_model):
        params_all = self.NN_model(inputs_NN)
        params_hydro_model = params_all[-1,:, :self.ny_hydro]
        params_temp_model = params_all[-1, :, self.ny_hydro: (self.ny_hydro + self.ny_temp)]
        params_PET_model = params_all[-1, :, (self.ny_hydro + self.ny_temp):]

        # Todo: I should separate PET model output from hydro_model and temp_model.
        #  For now, evap is calculated in both models individually (with same method)
        flowSim_total = self.hydro_model(
            x_hydro_model,
            c_hydro_model,
            params_hydro_model,
            self.args,
            PET_coef=params_PET_model,  # PET is in both temp and flow model
            warm_up=self.args["warm_up"],
            routing=self.args["routing_hydro_model"]
        )

        # Todo: send this to  a function
        # source flow calculation and converting mm/day to m3/ day
        srflow, ssflow, gwflow = self.hydro_model.source_flow_calculation(self.args, flowSim_total, c_hydro_model)

        temp_sim_all = self.temp_model.forward(x_temp_model,
                                                c_temp_model,
                                                params_temp_model,
                                                args=self.args,
                                                PET_coef=params_PET_model,
                                                srflow=srflow,
                                                ssflow=ssflow,
                                                gwflow=gwflow)

        BFI_sim = (torch.sum(gwflow, dim=0).squeeze() / (torch.sum(srflow + ssflow + gwflow, dim=0) + 0.00001).squeeze())[:, 0]
        PET_sim = torch.sum(temp_sim_all[5][:, :, 0],
                            dim=0) * 1000 * 86400  # the first one is PET , converting from m/s to mm/year

        return flowSim_total, temp_sim_all, BFI_sim, PET_sim











