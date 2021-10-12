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


def ave_temp_res_time(args, x, res_time, x_total, iGrid, iT):
    rho = args['hyperparameters']['rho']
    tArray_Total = tRange2Array(args['optData']['tRange'])
    tArray_train = tRange2Array(args['optData']['t_train'])
    _, ind1, _ = intersect(tArray_Total, tArray_train)
    resTrain = res_time[iGrid, :].cpu().int()
    # batchSize = iGrid.shape[0]
    # max_sr_res = torch.max(resTrain[:, 2].int()).item()
    # tTensor = torch.zeros([rho + max_sr_res, batchSize, 2], requires_grad=False)
    # tTensor[:, :, :] = float('nan')
    vars = args['optData']['varT'] + args['optData']['varC']
    # col_tmax = vars.index("tmax(C)")
    # col_tmin = vars.index("tmin(C)")
    # ind1_tensor = make_tensor(ind1, dtype=torch.int32, has_grad=False).repeat((res_time.shape[0], 1))
    ave_air_temp = make_tensor(np.full((resTrain.shape[0], rho, resTrain.shape[1]), np.nan), has_grad=False)
    for i in range(resTrain.shape[1]):
        for station in range(x.shape[1]):
            for day in range(x.shape[0]):
                tmax_temp = x_total[iGrid[station],
                            ind1[day].item() + iT[station] - resTrain[station, i].item() + 1: ind1[day].item() + iT[station] + 1,
                            vars.index("tmax(C)")].sum().item()
                tmin_temp = x_total[iGrid[station],
                            ind1[day].item() + iT[station] - resTrain[station, i].item() + 1: ind1[day].item() + iT[station] + 1,
                            vars.index("tmin(C)")].sum().item()
                ave_air_temp[station, day, i] = (tmax_temp + tmin_temp) / (2 * resTrain[station, i].item())
    return ave_air_temp, resTrain

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
    x_train, y_train, ngrid_train, nIterEp, nt, rho, batchSize = train_val_test_split("t_train", args, time1, x_total_raw, y_raw)

    # changing the numpy to tensor
    (x_total_raw_tensor, y_raw_tensor, c_raw_tensor,
     x_total_scaled_tensor, y_scaled_tensor, c_scaled_tensor,
     x_train_tensor, y_train_tensor) = make_tensor(x_total_raw, y_raw, c_raw,
                                                   x_total_scaled, y_scaled, c_scaled,
                                                   x_train, y_train)

    # ANN model to simulate parameters
    model = MLP(args)
    # loss function
    lossFun = crit.RmseLoss()
    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if torch.cuda.is_available():
        model = model.cuda()
        lossFun = lossFun.cuda()
        # moving dataset to CUDA

    T_w = STREAM_TEMP_EQ(args, x_total_raw_tensor)




    # training

    for epoch in range(1, args['hyperparameters']['EPOCHS'] + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(1, nIterEp + 1):
            res_time = model(c_scaled_tensor)

            iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho])
            xTrain = selectSubset(x_train, iGrid, iT, rho)
            ave_air_temp, resTrain = ave_temp_res_time(args, xTrain, res_time, x_total_raw, iGrid, iT)

            Yp = T_w.forward(xTrain.transpose(0, 1), c_raw_tensor, resTrain, ave_air_temp)
            iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho])
            # xTrain = selectSubset(x_total_raw, iGrid, iT, rho, c=C_total_raw)
            # Yp_train = selectSubset(Yp.unsqueeze(-1).cpu().detach().numpy(), iGrid, iT, rho)
            yObs = selectSubset(y_train, iGrid, iT, rho)
            loss = lossFun(Yp.unsqueeze(-1), yObs.transpose(1, 0))
            # loss.requires_grad = True
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
            print(iIter, " from ", nIterEp, " in the ", epoch, "th epoch, and Loss is ", loss.item())
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.6f} time {:.2f}'.format(
            epoch, lossEp,
            time.time() - t0)
        print(logStr)

        if epoch % args['hyperparameters']['saveEpoch'] == 0:
            # save model
            modelFile = os.path.join(args['output']['out_dir'],
                                     'model_Ep' + str(epoch) + '.pt')
            torch.save(model, modelFile)







    print('end')




if __name__=='__main__':
    args = config
    main(args)
