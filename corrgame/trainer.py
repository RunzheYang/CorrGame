## Online and Offline Training for General Correlation Game with Multi-Souce Inputs
## Created by Runzhe Yang on June 21, 2022

import numpy as np
from tqdm.notebook import tqdm, tnrange

# offline training
def train_offline(model, train_data, logger, init='ortho', mode='YWM', closed_form=None, niter=1000):
    for _ in tnrange(niter):
        model.learn(train_data, init=init, mode=mode, closed_form=closed_form)
        logger.log(model, train_data)

# online training        
def train_online(model, train_data, logger, init='ortho', mode='NN', 
                 closed_form=None, batch_size=1, epoch=1):
    for _ in tnrange(epoch):
        # multi-souce input
        if isinstance(train_data, list):
            T = train_data[0].size(1)
            randind = list(np.random.permutation(T))
            shuffled = [d[:,randind] for d in train_data]
            for k in tnrange(int(np.ceil(T/batch_size))):
                input_k = [s[:,batch_size*k:batch_size*(k+1)] for s in shuffled]
                model.learn(input_k, 
                            init=init, mode=mode, closed_form=closed_form)
                logger.log(model, input_k)
        else:
            T = train_data.size(1)
            randind = list(np.random.permutation(T))
            shuffled = train_data[:,randind]
            for k in tnrange(int(np.ceil(T/batch_size))):
                input_k = shuffled[:,batch_size*k:batch_size*(k+1)]
                model.learn(input_k, init=init, mode=mode, closed_form=closed_form)
                logger.log(model, input_k)