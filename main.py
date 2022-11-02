from core.read_configurations import config
from core.randomseed_config import randomseed_config
from core.data_prep import load_df, scaling, train_val_test_split, randomIndex, selectSubset
from core.small_codes import create_output_dirs
from MODELS.PGML_STemp import MLP, STREAM_TEMP_EQ, CudnnLstm, CudnnLstmModel
from MODELS import crit
from core import hydroDL
from core.small_codes import make_tensor, tRange2Array, intersect
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from post import stat, plot
import math
from ruamel.yaml import YAML


def syntheticP(args):
    x_total_temp, y_raw, c_raw = load_df(args)
    x_total_raw = x_total_temp.copy()
    x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp, y_raw, c_raw)
    time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
    x_train, y_train, ngrid_train, nIterEp, nt, rho, batchSize = train_val_test_split("t_train", args, time1,
                                                                                      x_total_raw, y_raw)
    x_train_scaled, y_train_scaled, _, _, _, _, _ = train_val_test_split("t_train", args, time1,
                                                                         x_total_scaled, y_scaled)
    x_total_raw_tensor = make_tensor(x_total_raw, has_grad=False)
    model = STREAM_TEMP_EQ(args, x_total_raw_tensor)
    if torch.cuda.is_available():
        # model = model.cuda()
        model = model.to(args["device"])
    # initializing the variables
    ave_air_temp = torch.empty((args['hyperparameters']['batch_size'],
                                x_train.shape[1],
                                len(args['params_target'])), device=args['device'], requires_grad=False)

    iGrid = np.arange(99)
    iT = np.zeros(99)
    Yp, ave_air_temp = model.forward(torch.from_numpy(x_train).cuda().float(), iGrid, iT, ave_air_temp)
    # Yp = model.forward(torch.from_numpy(x_train).cuda().float(), ave_air_temp)
    print('end')




def main(args):
    # setting random seeds
    # randomseed_config(args)
    mode_type = ["Meisner","SNTEMP"] #["van Vliet","Meisner","SNTEMP"]
    lenF_gwflow_list = [730, 730]
    lenF_ssflow_list = [1, 30]
    lat_temp_adj_list = ["True", "True"]
    frac_smoothening_list = ["True","False"]
    s = [0, 0]
    # seeds = args['randomseed']
    for seed, typ, LenF_gw, LenF_ss, adj,  frac_smooth in zip(s,
                                                            mode_type,
                                                            lenF_gwflow_list,
                                                            lenF_ssflow_list,
                                                            lat_temp_adj_list,
                                                            frac_smoothening_list):
        args["res_time_params"]["type"] = typ
        args["res_time_params"]["lenF_gwflow"] = LenF_gw
        args["res_time_params"]["lenF_ssflow"] = LenF_ss
        args["lat_temp_adj"] = adj
        # args["shade_smoothening"] = shade_smooth
        args["frac_smoothening"]["mode"] = frac_smooth
        # args['randomseed'] = seed
        # torch.cuda.set_per_process_memory_fraction(0.9)   # work for torch > 1.4
        randomseed_config(seed)
        # Creating output directories
        args = create_output_dirs(args, seed)
        min_max_scaler = preprocessing.MinMaxScaler()
        # getting the data
        x_total_temp, y_raw, c_raw = load_df(args)
        x_total_raw = x_total_temp.copy()
        x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp, y_raw, c_raw)
        c_scaled_0_1 = min_max_scaler.fit_transform(c_raw)
        time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
        #
        vars = args["optData"]["varT"] + args["optData"]["varC"]
        x_total_raw_tensor = make_tensor(x_total_raw, has_grad=False, device="cpu")


        # ANN model to simulate parameters
        # model = MLP(args)
        # all parameters are going to be 3D -> [batchsize, rho, nmul]
        no_tot_params = len(args["paramCalLst"])
        ny = args["nmul"] * no_tot_params
        model = CudnnLstmModel(nx=len(args["optData"]["varT"] + args["optData"]["varC"])-1,
                            ny=ny,
                            hiddenSize=args["hyperparameters"]["hidden_size"],
                            dr=args["hyperparameters"]["dropout"])
        Ts = STREAM_TEMP_EQ()
        # model = torch.load(r"/home/fzr5082/PGML_STemp_results/models/E_560_R_365_B_50_H_256_dr_0.5/model_Ep560.pt")
        #
        # loss function
        lossFun = crit.RmseLoss()
        optim = torch.optim.Adadelta(model.parameters())   # , lr=0.1
        # optim = torch.optim.SGD(model.parameters(), lr=10)

        if torch.cuda.is_available():
            # model = model.cuda()
            model = model.to(args["device"])
            # mlp = mlp.cuda()
            # Ts = Ts.cuda()
            Ts = Ts.to(args["device"])
            # lossFun = lossFun.cuda()
            lossFun = lossFun.to(args["device"])
            torch.backends.cudnn.deterministic = True
            CUDA_LAUNCH_BLOCKING = 1
            # moving dataset to CUDA

        c_tensorTrain = make_tensor(c_scaled_0_1, has_grad=False, device="cpu")




        if 0 in args['Action']:
            x_train, y_train, ngrid_train, nIterEp, nt, batchSize = train_val_test_split("t_train", args, time1,
                                                                                         x_total_raw, y_raw)
            x_train_scaled, y_train_scaled, _, _, _, _ = train_val_test_split("t_train", args, time1,
                                                                              x_total_scaled, y_scaled)

            vars = args["optData"]["varT"] + args["optData"]["varC"]
            x_train_scaled_noccov = np.delete(x_train_scaled, vars.index("ccov"), axis=2)

            ave_air_total = Ts.ave_temp_general(args, x_total_raw_tensor, time_range=args['optData']['t_train'])

            rho = args["hyperparameters"]["rho"]
            model.zero_grad()
            model.train()
            # training
            for epoch in range(1, args['hyperparameters']['EPOCHS'] + 1):
                lossEp = 0
                t0 = time.time()
                for iIter in range(1, nIterEp + 1):
                    iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho])

                    xTrain_sample = selectSubset(args, x_train, iGrid, iT, rho, has_grad=False)
                    xTrain_sample_scaled = selectSubset(args, x_train_scaled_noccov, iGrid, iT, rho, has_grad=False) #x_train_scaled

                    air_sample_sr = Ts.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_srflow'],
                                                            args=args, ave_air_total=ave_air_total)
                    air_sample_ss = Ts.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_ssflow'],
                                                            args=args, ave_air_total=ave_air_total)
                    air_sample_gw = Ts.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_gwflow'],
                                                            args=args, ave_air_total=ave_air_total)

                    ### MLP
                    if type(model) in [MLP]:
                        params = model(c_tensorTrain[iGrid])
                    ### CudnnLstm
                    if type(model) in [CudnnLstmModel]:
                        params = model(xTrain_sample_scaled.permute(1, 0, 2))
                    yObs = selectSubset(args, y_train, iGrid, iT, rho, has_grad=False)

                    Yp, ave_air_temp, gwflow_percentage, ssflow_percentage, gw_tau, ss_tau, pet, \
                    shade_fraction_riparian, shade_fraction_topo, \
                    top_width, cloud_fraction, hamon_coef, lat_temp_adj = Ts.forward(xTrain_sample.transpose(0, 1),
                                                                     params, iGrid, iT,
                                                                     args=args, air_sample_sr=air_sample_sr,
                                                                                     air_sample_ss=air_sample_ss,
                                                                                     air_sample_gw=air_sample_gw)

                    mask_yp = Yp.ge(1e-6)
                    y_sim = Yp * mask_yp.int().float()
                    loss = lossFun(y_sim.unsqueeze(-1), yObs.transpose(1, 0))
                    # loss = lossFun(test_sim, test)
                    # c = list(model.parameters())[0].clone()
                    loss.backward()  # retain_graph=True
                    # for param in model.parameters():
                    #     print(param.grad.data.sum())
                    optim.step()
                    # b = list(model.parameters())[0].clone()
                    # print(torch.equal(c.data, b.data))
                    model.zero_grad()
                    lossEp = lossEp + loss.item()

                    print(iIter, " from ", nIterEp, " in the ", epoch, "th epoch, and Loss is ", loss.item())
                lossEp = lossEp / nIterEp
                # torch.cuda.synchronize()
                logStr = 'Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory'.format(
                    epoch, lossEp,
                    time.time() - t0, int(torch.cuda.memory_allocated() * 0.001))
                print(logStr)

                if epoch % args['hyperparameters']['saveEpoch'] == 0:
                    # save model
                    modelFile = os.path.join(args['output']['out_dir'],
                                             'model_Ep' + str(epoch) + '.pt')
                    torch.save(model, modelFile)
                if epoch == args['hyperparameters']['EPOCHS']:
                    print('last epoch')
            print('end')
            del x_train_scaled, y_train_scaled, x_train_scaled_noccov, y_train, ave_air_total, x_train
            del gw_tau, ss_tau, pet, shade_fraction_riparian, shade_fraction_topo, \
                top_width, cloud_fraction, hamon_coef, lat_temp_adj

        if 1 in args['Action']:
            #to free up some GPU memory
            del x_total_temp, c_raw, y_scaled, c_scaled

            modelFile = os.path.join(args['output']['out_dir'],
                                      'model_Ep' + str(args['hyperparameters']['EPOCHS']) + '.pt')
            # modelFile = r"/home/fzr5082/PGML_STemp_results/models/415_sites/E_900_R_365_B_208_H_256_dr_0.5_0R1/model_Ep900.pt"

            model = torch.load(modelFile)
            model.eval()
            # iGrid = np.arange(99)

            time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
            x_test, y_test, ngrid_test, nIterEp, nt, batchSize = train_val_test_split("t_test", args, time1,
                                                                                              x_total_raw, y_raw)
            #Normalizing the inputs for ML part
            x_test_scaled, _, _, _, _, _ = train_val_test_split("t_test", args, time1,
                                                                                           x_total_scaled, y_raw)

            del x_total_raw, y_raw, x_total_scaled
            x_test_scaled_noccov = np.delete(x_test_scaled, vars.index("ccov"), axis=2)

            np.save(os.path.join(args['output']['out_dir'], 'x.npy'), x_test)
            x_test_tensor = make_tensor(x_test, has_grad=False)
            x_test_scaled_tensor = make_tensor(x_test_scaled_noccov, has_grad=False) #x_test_scaled
            y_test_tensor = make_tensor(y_test, has_grad=False)



            args_mod = args.copy()
            args_mod["hyperparameters"]["batch_size"] = args['no_basins']
            # args_mod["hyperparameters"]["rho"] = x_test_scaled_tensor.shape[1]
            ngrid, nt, nx = x_test_scaled_tensor.shape
            rho = args['hyperparameters']['rho']
            nrows = math.ceil(nt/rho)
            batch_size = args_mod["hyperparameters"]["batch_size"]
            iS = np.arange(0, ngrid, batch_size)
            iE = np.append(iS[1:], ngrid)
            ave_air_total = Ts.ave_temp_general(args, x_total_raw_tensor, time_range=args['optData']['t_test'])
            for i in range(0, len(iS)):
                # print('batch {}'.format(i))

                for j in range(nrows):
                    if j != (nrows - 1):
                        yTemp = torch.tensor(y_test_tensor[iS[i]:iE[i], j * rho: (j + 1) * rho, :])
                        xTemp_scaled = x_test_scaled_tensor[iS[i]:iE[i], j * rho: (j + 1) * rho, :]
                        xTemp = x_test_tensor[iS[i]:iE[i], j * rho: (j + 1) * rho, :]
                        ave_air_test = ave_air_total[iS[i]:iE[i], j * rho: (j + 1) * rho, :]

                        air_sample_sr = Ts.x_sample_air_temp2(iGrid=np.arange(0, ave_air_test.shape[0], 1),
                                                              iT=np.zeros(ave_air_test.shape[0]), lenF=args['res_time_params']['lenF_srflow'],
                                                              args=args, ave_air_total=ave_air_test)
                        air_sample_ss = Ts.x_sample_air_temp2(iGrid=np.arange(0, ave_air_test.shape[0], 1),
                                                              iT=np.zeros(ave_air_test.shape[0]), lenF=args['res_time_params']['lenF_ssflow'],
                                                              args=args, ave_air_total=ave_air_test)
                        air_sample_gw = Ts.x_sample_air_temp2(iGrid=np.arange(0, ave_air_test.shape[0], 1),
                                                              iT=np.zeros(ave_air_test.shape[0]), lenF=args['res_time_params']['lenF_gwflow'],
                                                              args=args, ave_air_total=ave_air_test)

                        if type(model) in [MLP]:
                            params = model(xTemp_scaled)
                        ### CudnnLstm
                        if type(model) in [CudnnLstmModel]:
                            params = model(xTemp_scaled)
                        iGrid = np.arange(xTemp.shape[0])
                        iT = np.zeros((len(iGrid)))
                        Yp, ave_air_temp, gwflow_percentage, ssflow_percentage, gw_tau, ss_tau, pet,\
                        shade_fraction_riparian, shade_fraction_topo, \
                        top_width, cloud_fraction, hamon_coef, lat_temp_adj = Ts.forward(xTemp, params, iGrid,
                                                      iT, args=args_mod, air_sample_sr=air_sample_sr,
                                                                                     air_sample_ss=air_sample_ss,
                                                                                     air_sample_gw=air_sample_gw)

                    else:
                        yTemp = torch.tensor(y_test_tensor[iS[i]:iE[i], j * rho:, :])
                        xTemp_scaled = x_test_scaled_tensor[iS[i]:iE[i], j * rho:, :]
                        xTemp = x_test_tensor[iS[i]:iE[i], j * rho:, :]
                        ave_air_test = ave_air_total[iS[i]:iE[i], j * rho:, :]
                        air_sample_sr = Ts.x_sample_air_temp2(iGrid=np.arange(0, ave_air_test.shape[0], 1),
                                                              iT=np.zeros(ave_air_test.shape[0]),
                                                              lenF=args['res_time_params']['lenF_srflow'],
                                                              args=args, ave_air_total=ave_air_test)
                        air_sample_ss = Ts.x_sample_air_temp2(iGrid=np.arange(0, ave_air_test.shape[0], 1),
                                                              iT=np.zeros(ave_air_test.shape[0]),
                                                              lenF=args['res_time_params']['lenF_ssflow'],
                                                              args=args, ave_air_total=ave_air_test)
                        air_sample_gw = Ts.x_sample_air_temp2(iGrid=np.arange(0, ave_air_test.shape[0], 1),
                                                              iT=np.zeros(ave_air_test.shape[0]),
                                                              lenF=args['res_time_params']['lenF_gwflow'],
                                                              args=args, ave_air_total=ave_air_test)
                        if type(model) in [MLP]:
                            params = model(xTemp_scaled)
                        ### CudnnLstm
                        if type(model) in [CudnnLstmModel]:
                            params = model(xTemp_scaled)
                        iGrid = np.arange(xTemp.shape[0])
                        iT = np.zeros((len(iGrid)))
                        Yp, ave_air_temp, gwflow_percentage, ssflow_percentage, gw_tau, ss_tau, pet,\
                        shade_fraction_riparian, shade_fraction_topo, \
                        top_width, cloud_fraction, hamon_coef, lat_temp_adj = Ts.forward(xTemp, params, iGrid,
                                                    iT, args=args_mod, air_sample_sr=air_sample_sr,
                                                                                     air_sample_ss=air_sample_ss,
                                                                                     air_sample_gw=air_sample_gw)

                    if (j == 0):
                        out = torch.clone(Yp.detach().cpu())
                        obstemp = torch.clone(yTemp)
                        gw = torch.clone(gwflow_percentage.unsqueeze(-1).detach().cpu())
                        ss = torch.clone(ssflow_percentage.unsqueeze(-1).detach().cpu())
                        w_gw_tau = torch.clone(gw_tau.unsqueeze(-1).detach().cpu())
                        w_ss_tau = torch.clone(ss_tau.unsqueeze(-1).detach().cpu())
                        PET = torch.clone(pet.unsqueeze(-1).detach().cpu())
                        shade_frac_rip = torch.clone(shade_fraction_riparian.unsqueeze(-1).detach().cpu())
                        shade_frac_top = torch.clone(shade_fraction_topo.unsqueeze(-1).detach().cpu())
                        top_w = torch.clone(top_width.unsqueeze(-1).detach().cpu())
                        cloud = torch.clone(cloud_fraction.unsqueeze(-1).detach().cpu())
                        hamon_co = torch.clone(hamon_coef.unsqueeze(-1).detach().cpu())
                        lat_temp = torch.clone(ave_air_temp.unsqueeze(-1).detach().cpu())
                        lat_temp_bias = torch.clone(lat_temp_adj.unsqueeze(-1).detach().cpu())
                    else:
                        out = torch.cat((out, Yp.detach().cpu()), dim=1)  # Farshid: should dim be 1 or 2?
                        obstemp = torch.cat((obstemp, yTemp), dim=1)
                        gw = torch.cat((gw, gwflow_percentage.unsqueeze(-1).detach().cpu()), dim=1)
                        ss = torch.cat((ss, ssflow_percentage.unsqueeze(-1).detach().cpu()), dim=1)
                        w_gw_tau = torch.cat((w_gw_tau, gw_tau.unsqueeze(-1).detach().cpu()), dim=1)
                        w_ss_tau = torch.cat((w_ss_tau, ss_tau.unsqueeze(-1).detach().cpu()), dim=1)
                        PET = torch.cat((PET, pet.unsqueeze(-1).detach().cpu()), dim=1)
                        shade_frac_rip = torch.cat((shade_frac_rip, shade_fraction_riparian.unsqueeze(-1).detach().cpu()), dim=1)
                        shade_frac_top = torch.cat((shade_frac_top, shade_fraction_topo.unsqueeze(-1).detach().cpu()), dim=1)
                        top_w = torch.cat((top_w, top_width.unsqueeze(-1).detach().cpu()), dim=1)
                        cloud = torch.cat((cloud, cloud_fraction.unsqueeze(-1).detach().cpu()), dim=1)
                        hamon_co = torch.cat((hamon_co, hamon_coef.unsqueeze(-1).detach().cpu()), dim=1)
                        lat_temp = torch.cat((lat_temp, ave_air_temp.unsqueeze(-1).detach().cpu()), dim=1)
                        lat_temp_bias = torch.cat((lat_temp_bias, lat_temp_adj.unsqueeze(-1).detach().cpu()), dim=1)
                if i == 0:
                    pred = torch.clone(out)
                    obs = torch.clone(obstemp)
                    gw_p = torch.clone(gw)
                    ss_p = torch.clone(ss)
                    weight_gw = torch.clone(w_gw_tau)
                    weight_ss = torch.clone(w_ss_tau)
                    PET_mm = torch.clone(PET)
                    shade_frac_rip_mm = torch.clone(shade_frac_rip)
                    shade_frac_top_mm = torch.clone(shade_frac_top)
                    top_w_mm = torch.clone(top_w)
                    cloud_mm = torch.clone(cloud)
                    hamon_co_mm = torch.clone(hamon_co)
                    lat_temp_mm = torch.clone(lat_temp)
                    lat_temp_bias_m = torch.clone(lat_temp_bias)
                else:
                    pred = torch.cat((pred, out), dim=0)
                    obs = torch.cat((obs, obstemp), dim=0)
                    gw_p = torch.cat((gw_p, gw), dim=0)
                    ss_p = torch.cat((ss_p, ss), dim=0)
                    weight_gw = torch.cat((weight_gw, w_gw_tau), dim=0)
                    weight_ss = torch.cat((weight_ss, w_ss_tau), dim=0)
                    PET_mm = torch.cat((PET_mm, PET), dim=0)
                    shade_frac_rip_mm = torch.cat((shade_frac_rip_mm, shade_frac_rip), dim=0)
                    shade_frac_top_mm = torch.cat((shade_frac_top_mm, shade_frac_top), dim=0)
                    top_w_mm = torch.cat((top_w_mm, top_w), dim=0)
                    cloud_mm = torch.cat((cloud_mm, cloud), dim=0)
                    hamon_co_mm = torch.cat((hamon_co_mm, hamon_co), dim=0)
                    lat_temp_mm = torch.cat((lat_temp_mm, lat_temp), dim=0)
                    lat_temp_bias_m = torch.cat((lat_temp_bias_m, lat_temp_bias), dim=0)




            mask_pred = pred.ge(0)
            y_sim = (pred * mask_pred.int().float()).unsqueeze(-1)
            loss = lossFun(y_sim.detach().cpu(), obs.detach().cpu())
            print(loss)


            predLst = list()
            obsLst = list()
            y_sim_np = y_sim.detach().cpu().numpy()
            y_obs_np = obs.detach().cpu().numpy()
            gw_p_np = gw_p.detach().cpu().numpy()
            ss_p_np = ss_p.detach().cpu().numpy()
            weight_gw_np = weight_gw.detach().cpu().numpy()
            weight_ss_np = weight_ss.detach().cpu().numpy()
            PET_mm_np = PET_mm.detach().cpu().numpy()
            shade_frac_rip_mm_np = shade_frac_rip_mm.detach().cpu().numpy()
            shade_frac_top_mm_np = shade_frac_top_mm.detach().cpu().numpy()
            top_w_mm_np = top_w_mm.detach().cpu().numpy()
            cloud_mm_np = cloud_mm.detach().cpu().numpy()
            hamon_co_mm_np = hamon_co_mm.detach().cpu().numpy()
            lat_temp_mm_np = lat_temp_mm.detach().cpu().numpy()
            lat_temp_bias_m_np = lat_temp_bias_m.detach().cpu().numpy()
            predLst.append(y_sim_np[:,365:,:])  # the prediction list for all the models
            obsLst.append(y_obs_np[:,365:,:])
            np.save(os.path.join(args['output']['out_dir'], 'pred.npy'), y_sim_np)
            np.save(os.path.join(args['output']['out_dir'], 'obs.npy'), y_obs_np)
            np.save(os.path.join(args['output']['out_dir'], 'gw_p.npy'), gw_p_np)
            np.save(os.path.join(args['output']['out_dir'], 'ss_p.npy'), ss_p_np)
            np.save(os.path.join(args['output']['out_dir'], 'weight_gw.npy'), weight_gw_np)
            np.save(os.path.join(args['output']['out_dir'], 'weight_ss.npy'), weight_ss_np)
            np.save(os.path.join(args['output']['out_dir'], 'PET.npy'), PET_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'shade_frac_rip.npy'), shade_frac_rip_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'shade_frac_topo.npy'), shade_frac_top_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'top_width.npy'), top_w_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'cloud_frac.npy'), cloud_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'hamon_coef.npy'), hamon_co_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'lat_temp.npy'), lat_temp_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'lat_temp_bias.npy'), lat_temp_bias_m_np)
            statDictLst = [stat.statError(x.squeeze(), y.squeeze()) for (x, y) in zip(predLst, obsLst)]
            ### save this file too
            # median and STD calculation
            count = 0
            mdstd = np.zeros([len(statDictLst[0]), 3])
            for i in statDictLst[0].values():
                median = np.nanmedian((i))  # abs(i)
                STD = np.nanstd((i))  # abs(i)
                mean = np.nanmean((i))  # abs(i)
                k = np.array([[median, STD, mean]])
                mdstd[count] = k
                count = count + 1
            mdstd = pd.DataFrame(mdstd, index=statDictLst[0].keys(), columns=['median', 'STD', 'mean'])
            mdstd.to_csv((os.path.join(args['output']['out_dir'], "mdstd.csv")))

            # Show boxplots of the results
            plt.rcParams['font.size'] = 14
            keyLst = ['Bias', 'RMSE', 'ubRMSE', 'NSE', 'Corr']
            dataBox = list()
            for iS in range(len(keyLst)):
                statStr = keyLst[iS]
                temp = list()
                for k in range(len(statDictLst)):
                    data = statDictLst[k][statStr]
                    data = data[~np.isnan(data)]
                    temp.append(data)
                dataBox.append(temp)
            labelname = ['PGML']  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']

            xlabel = ['Bias ($\mathregular{deg}$C)', 'RMSE', "ubRMSE", 'NSE', 'Corr']
            fig = plot.plotBoxFig(dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8))
            fig.patch.set_facecolor('white')
            boxPlotName = "PGML"
            fig.suptitle(boxPlotName, fontsize=12)
            plt.rcParams['font.size'] = 12
            plt.savefig(os.path.join(args['output']['out_dir'], "Boxplot.png"))  # , dpi=500
            fig.show()
            plt.close()
            print("END testing")
            del x_total_raw_tensor, x_test_scaled_noccov, x_test_scaled
            del gw_tau, ss_tau, pet, shade_fraction_riparian, shade_fraction_topo, \
                top_width, cloud_fraction, hamon_coef, lat_temp_adj
            torch.cuda.empty_cache()




if __name__=='__main__':
    args = config
    # syntheticP(args)
    main(args)
    print('END')












