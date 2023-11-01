import os
import numpy as np
import pandas as pd
import torch
import math
import time
from post import stat, plot
import matplotlib.pyplot as plt
from core.load_data.dataFrame_loading import loadData
from core.load_data.data_prep import (
    No_iter_nt_ngrid,
    randomIndex,
    selectSubset,
)
from core.load_data.normalizing import transNorm
def train_differentiable_model(args, diff_model, lossFun, optim):
    if torch.cuda.is_available():
        diff_model = diff_model.to(args["device"])
        lossFun = lossFun.to(args["device"])
        torch.backends.cudnn.deterministic = True
        CUDA_LAUNCH_BLOCKING = 1

    # preparing training dataset
    x_NN, c_NN, y_obs, x_hydro_model, c_hydro_model, x_temp_model, c_temp_model = loadData(args, trange=args["t_train"])
    # normalizing
    x_NN_scaled = transNorm(args, x_NN, varLst=args["varT_NN"], toNorm=True)
    c_NN_scaled = transNorm(args, c_NN, varLst=args["varC_NN"], toNorm=True)
    c_NN_scaled2 = np.repeat(np.expand_dims(c_NN_scaled, 1), x_NN.shape[1]).reshape(x_NN.shape[0], x_NN.shape[1], c_NN.shape[1])
    inputs_NN_model = np.concatenate((x_NN_scaled, c_NN_scaled2), axis=2)

    ngrid_train, nIterEp, nt, batchSize = No_iter_nt_ngrid("t_train", args, inputs_NN_model)
    rho = args["rho"]
    warm_up = args["warm_up"]
    nmul = args["nmul"]
    diff_model.zero_grad()
    diff_model.train()
    # training
    for epoch in range(1, args["EPOCHS"] + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(1, nIterEp + 1):
            iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho + warm_up])

            # NN sampling
            x_NN_sample_scaled = selectSubset(
                args, inputs_NN_model, iGrid, iT, rho + warm_up, has_grad=False,
            )
            # Hydro model sampling
            x_hydro_model_sample = selectSubset(
                args, x_hydro_model, iGrid, iT, rho + warm_up, has_grad=False
            )
            c_hydro_model_sample = torch.tensor(
                c_hydro_model[iGrid], device=args["device"], dtype=torch.float32
            )
            # temperture model sampling
            x_temp_model_sample = selectSubset(
                args, x_temp_model, iGrid, iT, rho + warm_up, has_grad=False
            )  # [warm_up:,:, :]there is no need for warm up in temp section yet
            c_temp_model_sample = torch.tensor(
                c_temp_model[iGrid], device=args["device"], dtype=torch.float32
            )
            # Batch running of the differentiable model
            out_diff_model = diff_model(x_NN_sample_scaled,
                                        x_hydro_model_sample, c_hydro_model_sample,
                                        x_temp_model_sample, c_temp_model_sample)
            # collecting observation samples
            obs_sample = selectSubset(
                args, y_obs, iGrid, iT, rho + warm_up, has_grad=False
            )[warm_up:, :, :]
            obs_sample = converting_flow_from_ft3_per_sec_to_mm_per_day(args, c_hydro_model_sample, obs_sample)
            loss = lossFun(args, out_diff_model, obs_sample)
            loss.backward()  # retain_graph=True
            optim.step()
            diff_model.zero_grad()
            lossEp = lossEp + loss.item()
            if (iIter % 1 == 0) or (iIter == nIterEp):
                print(iIter, " from ", nIterEp, " in the ", epoch,
                      "th epoch, and Loss is ", loss.item())
        lossEp = lossEp / nIterEp
        logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory".format(
            epoch, lossEp, time.time() - t0,
            int(torch.cuda.memory_allocated(device=args["device"]) * 0.001))
        print(logStr)

        if epoch % args["saveEpoch"] == 0:
            # save model
            modelFile = os.path.join(args["out_dir"], "model_Ep" + str(epoch) + ".pt")
            torch.save(diff_model, modelFile)
        if epoch == args["EPOCHS"]:
            print("last epoch")
    print("Training ended")

def converting_flow_from_ft3_per_sec_to_mm_per_day(args, c_hydro_model_sample, obs_sample):
    varTar_NN = args["target"]
    obs_flow_v = obs_sample[:, :, varTar_NN.index("00060_Mean")]
    varC_hydro_model = args["varC_hydro_model"]
    if "DRAIN_SQKM" in varC_hydro_model:
        area_name = "DRAIN_SQKM"
    elif "area_gages2" in varC_hydro_model:
        area_name = "area_gages2"
    area = (c_hydro_model_sample[:, varC_hydro_model.index(area_name)]).unsqueeze(0).repeat(obs_flow_v.shape[0], 1)
    obs_sample[:, :, varTar_NN.index("00060_Mean")] = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6)) # convert ft3/s to mm/day
    return obs_sample
def test_differentiable_model(args, diff_model):
    warm_up = args["warm_up"]
    nmul = args["nmul"]
    diff_model.eval()
    # read data for test time range
    x_NN, c_NN, y_obs, x_hydro_model, c_hydro_model, x_temp_model, c_temp_model = loadData(args, trange=args["t_test"])
    np.save(os.path.join(args["out_dir"], "x.npy"), x_NN)  # saves with the overlap in the beginning
    # normalizing
    x_NN_scaled = transNorm(args, x_NN, varLst=args["varT_NN"], toNorm=True)
    c_NN_scaled = transNorm(args, c_NN, varLst=args["varC_NN"], toNorm=True)
    c_NN_scaled2 = np.repeat(np.expand_dims(c_NN_scaled, 1), x_NN.shape[1]).reshape(x_NN.shape[0], x_NN.shape[1],
                                                                                    c_NN.shape[1])
    inputs_NN_model = np.concatenate((x_NN_scaled, c_NN_scaled2), axis=2)
    # converting the numpy arrays to torch tensors:
    inputs_NN_model_T = torch.from_numpy(np.swapaxes(inputs_NN_model, 1, 0)).float().to(args["device"])
    x_hydro_model_T = torch.from_numpy(np.swapaxes(x_hydro_model, 1, 0)).float().to(args["device"])
    c_hydro_model_T = torch.from_numpy(c_hydro_model).float().to(args["device"])
    x_temp_model_T = torch.from_numpy(np.swapaxes(x_temp_model, 1, 0)).float().to(args["device"])
    c_temp_model_T = torch.from_numpy(c_temp_model).float().to(args["device"])
    args_mod = args.copy()
    args_mod["batch_size"] = args["no_basins"]
    ngrid, nt, nx = inputs_NN_model.shape
    rho = args["rho"]
    nrows = math.ceil((nt - warm_up) / rho)  # need to reduce the warm_up from beginning
    batch_size = args["batch_size"]
    iS = np.arange(0, ngrid, batch_size)
    iE = np.append(iS[1:], ngrid)
    list_out_diff_model = []
    for i in range(0, len(iS)):
        x_NN_sample_scaled = inputs_NN_model_T[:, iS[i]: iE[i], :]
        x_hydro_model_sample = x_hydro_model_T[:, iS[i]: iE[i], :].type(torch.float32)
        c_hydro_model_sample = c_hydro_model_T[iS[i]: iE[i], :]
        x_temp_model_sample = x_temp_model_T[:, iS[i]: iE[i], :]
        c_temp_model_sample = c_temp_model_T[iS[i]: iE[i], :]
        out_diff_model = diff_model(x_NN_sample_scaled,
                                    x_hydro_model_sample,
                                    c_hydro_model_sample,
                                    x_temp_model_sample,
                                    c_temp_model_sample)
        # Convert all tensors in the dictionary to CPU
        out_diff_model_cpu = {key: tensor.cpu().detach() for key, tensor in out_diff_model.items()}
        # out_diff_model_cpu = tuple(outs.cpu().detach() for outs in out_diff_model)
        list_out_diff_model.append(out_diff_model_cpu)

    # getting rid of warm-up period in observation dataset and making the dimension similar to
    # converting numpy to tensor
    y_obs = torch.tensor(np.swapaxes(y_obs[:, warm_up:, :], 0, 1), dtype=torch.float32)
    c_hydro_model = torch.tensor(c_hydro_model, dtype=torch.float32)
    y_obs = converting_flow_from_ft3_per_sec_to_mm_per_day(args, c_hydro_model, y_obs)

    save_outputs(args, list_out_diff_model, y_obs, calculate_metrics=True)
    torch.cuda.empty_cache()
    print("Testing ended")
def save_outputs(args, list_out_diff_model, y_obs, calculate_metrics=True):
    for key in list_out_diff_model[0].keys():
        if len(list_out_diff_model[0][key].shape) == 3:
            dim = 1
        if len(list_out_diff_model[0][key].shape) == 1:
            dim = 0
        concatenated_tensor = torch.cat([d[key] for d in list_out_diff_model], dim=dim)
        file_name = key + ".npy"
        np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), concatenated_tensor.numpy())

    # Reading flow observation
    for var in args["target"]:
        item_obs = y_obs[:, :, args["target"].index(var)]
        file_name = var + ".npy"
        np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), item_obs)

    if calculate_metrics == True:
        predLst = list()
        obsLst = list()
        flow_sim = torch.cat([d["flow_sim"] for d in list_out_diff_model], dim=1)
        flow_obs = y_obs[:, :, args["target"].index("00060_Mean")]
        predLst.append(flow_sim.numpy())
        obsLst.append(np.expand_dims(flow_obs, 2))
        if args["temp_model_name"] != "None":
            temp_sim = torch.cat([d["temp_sim"] for d in list_out_diff_model], dim=1)
            temp_obs = y_obs[:, :, args["target"].index("00010_Mean")]
            predLst.append(temp_sim.numpy())
            obsLst.append(np.expand_dims(temp_obs, 2))
        statDictLst = [
            stat.statError(x.squeeze(), y.squeeze())
            for (x, y) in zip(predLst, obsLst)
        ]
        ### save this file
        # median and STD calculation
        name_list = ["flow", "temp"]
        for st, name in zip(statDictLst, name_list):
            count = 0
            mdstd = np.zeros([len(st), 3])
            for key in st.keys():
                median = np.nanmedian(st[key])  # abs(i)
                STD = np.nanstd(st[key])  # abs(i)
                mean = np.nanmean(st[key])  # abs(i)
                k = np.array([[median, STD, mean]])
                mdstd[count] = k
                count = count + 1
            mdstd = pd.DataFrame(
                mdstd, index=st.keys(), columns=["median", "STD", "mean"]
            )
            mdstd.to_csv((os.path.join(args["out_dir"], args["testing_dir"], "mdstd_" + name + ".csv")))

            # Show boxplots of the results
            plt.rcParams["font.size"] = 14
            keyLst = ["Bias", "RMSE", "ubRMSE", "NSE", "Corr"]
            dataBox = list()
            for iS in range(len(keyLst)):
                statStr = keyLst[iS]
                temp = list()
                # for k in range(len(st)):
                data = st[statStr]
                data = data[~np.isnan(data)]
                temp.append(data)
                dataBox.append(temp)
            labelname = [
                "Hybrid differentiable model"
            ]  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']

            xlabel = ["Bias ($\mathregular{deg}$C)", "RMSE", "ubRMSE", "NSE", "Corr"]
            fig = plot.plotBoxFig(
                dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8)
            )
            fig.patch.set_facecolor("white")
            boxPlotName = "PGML"
            fig.suptitle(boxPlotName, fontsize=12)
            plt.rcParams["font.size"] = 12
            plt.savefig(
                os.path.join(args["out_dir"], args["testing_dir"], "Box_" + name + ".png")
            )  # , dpi=500
            # fig.show()
            plt.close()










