from core.read_configurations import config
from core import randomseed_config
from core.data_prep import load_df, scaling, train_val_test_split
from core.small_codes import create_output_dirs
from MODELS.PGML_STemp import PGML_STemp, STREAM_TEMP_EQ
from core import hydroDL
import torch
import pandas as pd

def main(args):
    # setting random seeds
    randomseed_config

    # getting the data
    x_total_raw, y_total_raw, C_total_raw = load_df(args)
    x_total_scaled, y_total_scaled = scaling(args, x_total_raw, y_total_raw)
    time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
    x, y, ngrid_train, nIterEp, nt = train_val_test_split("t_train", args, time1, x_total_scaled, y_total_scaled)

    # Creating output directories
    args = create_output_dirs(args)
    model = PGML_STemp
    if torch.cuda.is_available():
        model = model.cuda()





    print('end')




if __name__=='__main__':
    args = config
    main(args)
