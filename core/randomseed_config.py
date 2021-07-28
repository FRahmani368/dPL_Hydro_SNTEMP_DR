import numpy as np
import torch
import random
from core.read_configurations import config
if config['randomseed'] is None:
    # generate random seed
    randomseed = int(np.random.uniform(low=0, high=1e6))
    optTrain['seed'] = randomseed
    print('random seed updated!')
else:
    randomseed = config['randomseed']
    random.seed(randomseed)
    torch.manual_seed(randomseed)
    np.random.seed(randomseed)
    torch.cuda.manual_seed(randomseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False