import numpy as np
import torch
import random


def randomseed_config(args):
    if args['randomseed'] is None:
        # generate random seed
        randomseed = int(np.random.uniform(low=0, high=1e6))
        optTrain['seed'] = randomseed
        print('random seed updated!')
    else:
        randomseed = args['randomseed']
        random.seed(randomseed)
        torch.manual_seed(randomseed)
        np.random.seed(randomseed)
        torch.cuda.manual_seed(randomseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False