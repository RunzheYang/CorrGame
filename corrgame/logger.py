## Training Logger for General Correlation Game with Multi-Souce Inputs
## Created by Runzhe Yang on June 21, 2022

class Logger(object):
    def __init__(self, evals):
        self.evals = evals
        self.history = {e: [] for e in evals.keys()}
        
    def log(self, model, train_data):
        for e in self.evals.keys():
            self.history[e].append(self.evals[e](model, train_data))
    
    def clear(self):
        self.history = {e: [] for e in evals.keys()}