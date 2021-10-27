from core.read_configurations import config
from core.randomseed_config import randomseed_config
from core.data_prep import load_df, scaling, train_val_test_split, randomIndex, selectSubset
from core.small_codes import create_output_dirs
from MODELS.PGML_STemp import MLP, STREAM_TEMP_EQ
from MODELS import crit
from core import hydroDL
from core.small_codes import make_tensor, tRange2Array, intersect
import torch
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


def ave_temp_res_time(args, ave_air_temp, ave_air, x, res_time, x_total, iGrid, iT):
    rho = x.shape[0] #args['hyperparameters']['rho']
    tArray_Total = tRange2Array(args['optData']['tRange'])
    tArray_train = tRange2Array(args['optData']['t_train'])
    _, ind1, _ = (intersect(tArray_Total, tArray_train))
    ind1_tensor = make_tensor(ind1, has_grad=False)
    iT_tensor = make_tensor(iT, has_grad=False)
    vars = args['optData']['varT'] + args['optData']['varC']
    temp_res = res_time
    with torch.no_grad():
        temp_res1 = temp_res.int()
    A = res_time.repeat(1, rho)
    B = torch.reshape(A, (res_time.shape[0], rho, res_time.shape[1]))
    for i in range(res_time.shape[1]):
        for station in range(x.shape[1]):
            array = np.zeros((x.shape[0], temp_res1[station, i].item()), dtype=np.int32)
            for j in range(temp_res1[station, i].item()):
                array[:, j] = np.arange((ind1_tensor[0] + iT_tensor[station] - j).item(),
                                           (ind1_tensor[0] + iT_tensor[station] - j + x.shape[0]).item())
            tmax_temp = x_total[station, array, vars.index("tmax(C)")]
            max_add = torch.sum(tmax_temp, dim=1)
            tmin_temp = x_total[station, array, vars.index("tmin(C)")]
            min_add = torch.sum(tmin_temp, dim=1)
            ave_air[station, :, i] = (max_add + min_add) / 2 #(2 * res_time[station, i])
    ave_air_temp = ave_air / B
    # return ave_air
    return ave_air_temp, ave_air

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
    T_w = STREAM_TEMP_EQ(args, x_total_raw_tensor)

    # initializing the variables
    ave_air_temp = torch.empty((args['hyperparameters']['batch_size'],
                                x_train.shape[1],
                                len(args['params_target'])), device=args['device'], requires_grad=False)

    ave_air = torch.empty((args['hyperparameters']['batch_size'],
                           x_train.shape[1],
                           len(args['params_target'])), device=args['device'], requires_grad=False)


    factor = torch.empty((args['hyperparameters']['batch_size'],
                          len(args['params_target'])), device=args['device'], requires_grad=False)
    factor[:, 0] = 1  # srflow residence time
    factor[:, 1] = 10  # ssflow residence time
    factor[:, 2] = 365  # gwflow residence factor
    synthP = torch.zeros((99, 3), device=args['device'])
    synthP_mod = synthP + factor

    ave_air_temp, ave_air = ave_temp_res_time(args, ave_air_temp, ave_air, torch.from_numpy(x_train.transpose(1, 0, 2)), synthP_mod,
                                              x_total_raw_tensor, 0, np.zeros(synthP_mod.shape[0]))
    #
    Yp = T_w.forward(torch.from_numpy(x_train).cuda().float(), ave_air_temp)
    print('end')




def main(args):
    # setting random seeds
    randomseed_config(args)

    # Creating output directories
    args = create_output_dirs(args)

    # getting the data
    x_total_temp, y_raw, c_raw = load_df(args)
    x_total_raw = x_total_temp.copy()
    x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp, y_raw, c_raw)
    time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
    x_train, y_train, ngrid_train, nIterEp, nt, rho, batchSize = train_val_test_split("t_train", args, time1,
                                                                                      x_total_raw, y_raw)
    x_train_scaled, y_train_scaled, _, _, _, _, _ = train_val_test_split("t_train", args, time1,
                                                                                      x_total_scaled, y_scaled)


    # changing the numpy to tensor
    # (x_total_raw_tensor, y_raw_tensor, c_raw_tensor,
    #  x_total_scaled_tensor, y_scaled_tensor, c_scaled_tensor,
    #  x_train_tensor, y_train_tensor) = make_tensor(x_total_raw, y_raw, c_raw,
    #                                                x_total_scaled, y_scaled, c_scaled,
    #                                                x_train, y_train)
    x_total_raw_tensor = make_tensor(x_total_raw, has_grad=False)

    # ANN model to simulate parameters
    model = MLP(args)
    # loss function
    lossFun = crit.RmseLoss()
    optim = torch.optim.Adadelta(model.parameters())
    # optim = torch.optim.SGD(model.parameters(), lr=1)

    if torch.cuda.is_available():
        model = model.cuda()
        lossFun = lossFun.cuda()
        # moving dataset to CUDA
    model.zero_grad()
    T_w = STREAM_TEMP_EQ(args, x_total_raw_tensor)

    # initializing the variables
    ave_air_temp = torch.empty((args['hyperparameters']['batch_size'],
                                args['hyperparameters']['rho'],
                                len(args['params_target'])), device=args['device'], requires_grad=False)
    ave_air = torch.empty((args['hyperparameters']['batch_size'],
                                args['hyperparameters']['rho'],
                                len(args['params_target'])), device=args['device'], requires_grad=False)
    ave_air_temp23 = torch.empty((args['hyperparameters']['batch_size'],
                                  x_train.shape[1],
                                  2), device=args['device'], requires_grad=False)
    ave_air23 = torch.empty((args['hyperparameters']['batch_size'],
                             args['hyperparameters']['rho'],
                             2), device=args['device'], requires_grad=False)
    res23 = torch.empty((args['hyperparameters']['batch_size'],
                             2), device=args['device'], requires_grad=False)

    factor = torch.empty((args['hyperparameters']['batch_size'],
                                len(args['params_target'])), device=args['device'], requires_grad=False)
    factor[:, 0] = 10   # factor for srflow residence time
    res23[:, 0] = 10.0   # value of ssflow residence time
    res23[:, 1] = 365.0   # value of gwflow residence factor
    # training

    for epoch in range(1, args['hyperparameters']['EPOCHS'] + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(1, nIterEp + 1):
            iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho])
            xTrain_sample = selectSubset(x_train, iGrid, iT, rho, has_grad=False)
            xTrain_sample_scaled = selectSubset(x_train_scaled, iGrid, iT, rho, has_grad=False)
            yObs = selectSubset(y_train, iGrid, iT, rho, has_grad=False)
            c_tensorTrain = make_tensor(c_scaled[iGrid], has_grad=False)
            res_time = model(c_tensorTrain)
            res_time_mod = res_time * factor + 1 #.round()

            # res_time = model(xTrain_sample_scaled)
            ave_air_temp, ave_air = ave_temp_res_time(args, ave_air_temp, ave_air, xTrain_sample, res_time_mod, x_total_raw_tensor[iGrid], iGrid, iT)
            ave_air_temp23, ave_air23 = ave_temp_res_time(args, ave_air_temp23, ave_air23, xTrain_sample, res23,
                                                      x_total_raw_tensor[iGrid], iGrid, iT)
            Yp = T_w.forward(xTrain_sample.transpose(0, 1), ave_air_temp, ave_air_temp23)
            loss = lossFun(Yp.unsqueeze(-1), yObs.transpose(1, 0))
            # c = list(model.parameters())[0].clone()
            loss.backward()  # retain_graph=True
            # for param in model.parameters():
            #     print(param.grad.data.sum())
            optim.step()
            # b = list(model.parameters())[0].clone()
            # print(torch.equal(c.data, b.data))
            model.zero_grad()
            lossEp = lossEp + loss.item()
            # del loss
            # del Yp
            # print(iIter, " from ", nIterEp, " in the ", epoch, "th epoch, and Loss is ", loss.item())
        lossEp = lossEp / nIterEp
        # torch.cuda.synchronize()
        logStr = 'Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory'.format(
            epoch, lossEp,
            time.time() - t0, int(torch.cuda.memory_allocated() * 0.001))
        print(logStr)
        if lossEp < 0.05:
            torch.save(res_time, os.path.join(args['output']['data'], 'res_time' + str(lossEp) + ".pt"))
            torch.save(res_time_mod, os.path.join(args['output']['data'], 'res_time_mod' + str(lossEp) + ".pt"))
        elif lossEp != lossEp:
            print("stop")
        if epoch % args['hyperparameters']['saveEpoch'] == 0:
            # save model
            modelFile = os.path.join(args['output']['out_dir'],
                                     'model_Ep' + str(epoch) + '.pt')
            torch.save(model, modelFile)

    print('end')

if __name__=='__main__':
    args = config
    # syntheticP(args)
    main(args)
    print('END')
