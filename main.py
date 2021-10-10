from core.read_configurations import config
from core import randomseed_config
from core.data_prep import load_df, scaling, train_val_test_split, randomIndex, selectSubset
from core.small_codes import create_output_dirs
from MODELS.PGML_STemp import MLP, STREAM_TEMP_EQ
from MODELS import crit
from core import hydroDL
from core.small_codes import make_tensor
import torch
import numpy as np
import pandas as pd
import time
import os

def main(args):
    # setting random seeds
    randomseed_config

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

    for epoch in range(args['hyperparameters']['EPOCHS']):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            res_time = model(c_scaled_tensor)
            Yp = T_w.forward(x_train_tensor, c_raw_tensor, res_time)
            iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho])
            # xTrain = selectSubset(x_total_raw, iGrid, iT, rho, c=C_total_raw)
            Yp_train = selectSubset(Yp.unsqueeze(-1).cpu().detach().numpy(), iGrid, iT, rho)
            yObs = selectSubset(y_train_tensor.cpu().detach().numpy(), iGrid, iT, rho)
            loss = lossFun(Yp_train.transpose(1, 0), yObs.transpose(1, 0))
            loss.requires_grad = True
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
