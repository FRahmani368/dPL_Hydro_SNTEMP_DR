import torch
import json
import os

class RmseLoss_flow_temp(torch.nn.Module):
    def __init__(self, w1=0.5, w2=None, alpha=0.25, beta=1e-6):
        super(RmseLoss_flow_temp, self).__init__()
        self.w1 = w1
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.w2 = w2

    def forward(self, args, c_hydro_model_sample, y_obs, y_sim):
        if self.w2 == None:    # w1 + w2 =1
            self.w2 = 1 - self.w1

        varTar_NN = args["target"]
        obs_flow_v = y_obs[:, :, varTar_NN.index("00060_Mean")]
        obs_temp = y_obs[:, :, varTar_NN.index("00010_Mean")]
        sim_flow = y_sim[0][:, :, 0]  # simulation
        sim_temp = y_sim[1][0]
        varC_hydro_model = args["varC_hydro_model"]
        if "DRAIN_SQKM" in varC_hydro_model:
            area_name = "DRAIN_SQKM"
        elif "area_gage2" in varC_hydro_model:
            area_name = "area_gage2"
        area = c_hydro_model_sample[:, varC_hydro_model.index(area_name)].unsqueeze(0).repeat(obs_flow_v.shape[0], 1)
        obs_flow = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6))  # convert ft3/s to mm/day
        # flow
        if len(obs_flow[obs_flow==obs_flow]) > 0:
            mask_flow1 = obs_flow == obs_flow
            p = sim_flow[mask_flow1]
            t = obs_flow[mask_flow1]
            loss_flow1 = torch.sqrt(((p - t) ** 2).mean())  # RMSE item

            p1 = torch.log10(torch.sqrt(sim_flow + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(obs_flow + self.beta) + 0.1)
            mask_flow2 = t1 == t1
            pa = p1[mask_flow2]
            ta = t1[mask_flow2]
            loss_flow2 = torch.sqrt(((pa - ta) ** 2).mean())  # Log-Sqrt RMSE item
            loss_flow_total = (1.0-self.alpha) * loss_flow1 + self.alpha * loss_flow2
        else:
            loss_flow_total = 0.0

        # temp
        if len(obs_temp[obs_temp==obs_temp]) > 0:
            mask_temp1 = obs_temp == obs_temp
            p_temp = sim_temp[mask_temp1]
            t_temp = obs_temp[mask_temp1]
            loss_temp = torch.sqrt(((p_temp - t_temp) ** 2).mean())  # RMSE item
        else:
            loss_temp = 0.0