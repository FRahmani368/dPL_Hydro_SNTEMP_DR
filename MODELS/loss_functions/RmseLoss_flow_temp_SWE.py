import torch
import json
import os

class RmseLoss_flow_temp_SWE(torch.nn.Module):
    def __init__(self, w1=5, w2=None,  w3=0.05, alpha=0.25, beta=1e-6):
        super(RmseLoss_flow_temp_SWE, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta

    def forward(self, args, y_sim, y_obs, igrid):
        if self.w2 == None:    # w1 + w2 =1
            self.w2 = 1 - self.w1
        # flow
        varTar_NN = args["target"]
        obs_flow = y_obs[:, :, varTar_NN.index("00060_Mean")]
        obs_temp = y_obs[:, :, varTar_NN.index("00010_Mean")]
        obs_SWE = y_obs[:, :, varTar_NN.index("swe_nsidc")]

        sim_flow = y_sim["flow_sim"].squeeze()    #  simulation
        sim_temp = y_sim["temp_sim"].squeeze()
        sim_SWE = y_sim["SWE_sim"].squeeze()

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

        # SWE
        if len(obs_SWE[obs_SWE == obs_SWE]) > 0:
            mask_SWE1 = obs_SWE == obs_SWE
            p_SWE = sim_SWE[mask_SWE1]
            t_SWE = obs_SWE[mask_SWE1]
            p_SWE2 = torch.where((torch.abs((p_SWE - t_SWE)) / t_SWE > 0.1), p_SWE, t_SWE)

            loss_SWE = torch.sqrt(((p_SWE2 - t_SWE) ** 2).mean())  # RMSE item
        else:
            loss_SWE = 0.0

        loss = self.w1 * loss_flow_total + self.w2 * loss_temp + self.w3 * loss_SWE
        return loss
