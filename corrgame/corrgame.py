## General Correlation Game with Multi-Souce Inputs
## Created by Runzhe Yang on June 21, 2022

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# use gpu acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from functools import partial
from scipy.stats import ortho_group
from typing import Iterable, Dict

class CorrGame(nn.Module):
    def __init__(self, n, k, Phi, Psi, 
                 constraints = None, eta=None, dPhi=None, dPsi=None, device='cpu'):
        super(CorrGame, self).__init__()
        self.n, self.k = n, k
        self.multi_inputs = False
        if isinstance(self.n, Iterable):
            self.multi_inputs = True
            self.W = [torch.Tensor(k, nx).normal_(0,1e-2).to(device) for nx in self.n]
        else:
            self.W = torch.Tensor(k, n).normal_(0,1e-2).to(device)
        self.M = torch.Tensor(k, k).normal_(0,1e-2).to(device)
        self.M = self.M.mm(self.M.t())
        self.Y = None
        self.Phi = Phi
        self.Psi = Psi
        
        self.proj = constraints if constraints else {var: lambda x:x for var in ['Y', 'W', 'M']}
        
        self.dPhi = dPhi if dPhi else self.derivative(self.Phi)
        self.dPsi = dPsi if dPsi else self.derivative(self.Psi)
        
        self.eta_W = eta['W'] if eta else 0.1
        self.eta_M = eta['M'] if eta else 0.05
        self.eta_Y = eta['Y'] if eta else 0.01

        self.device = device
    
    # auto derivative if not given
    def derivative(self, fn):
        def df(fn, w, x):
            w = Variable(w, requires_grad=True)
            y = fn(w, x)
            y.backward()
            return w.grad
        return partial(df, fn)
    
    # correlation AB' / T
    def corr(self, A, B, normalize=True):
        return A.mm(B.t()) / A.size(1)
    
    # offline or online updates
    def learn(self, X, mode='YWM', init='ortho', closed_form=None, errs=None, step_limit=1000):
        
        # initialize Y
        if self.Y is None: 
            if self.multi_inputs:
                self.Y = 0
                for i, nx in enumerate(self.n):
                    if init == 'ortho':
                        init_filter = ortho_group.rvs(nx)[:self.k]
                        init_filter = torch.Tensor(init_filter).to(self.device)
                        self.Y += init_filter.mm(X[i])
                        self.W[i] = self.M.mm(init_filter)
                    elif init == 'whiten':
                        init_filter = ortho_group.rvs(nx)[:self.k]
                        init_filter = torch.Tensor(init_filter).to(self.device)
                        U_x, S_x, Vh_x = torch.svd(X[i].mm(X[i].t())/X[i].size(1))
                        init_filter = init_filter.mm(U_x.mm(torch.diag(1/torch.sqrt(S_x))).mm(U_x.t()))
                        self.Y += init_filter.mm(X[i])
                        self.W[i] = self.M.mm(init_filter)
                    elif isinstance(init, Dict):
                        self.Y = init['Y']
                        self.W = init['W']
                        self.M = init['M']
                        break
                    self.W[i] = self.proj['W'](self.W[i])
                self.Y = self.proj['Y'](self.Y)
                self.M = self.proj['M'](self.M)
            else:
                if init == 'ortho': 
                    init_filter = ortho_group.rvs(self.n)[:self.k]
                    init_filter = torch.Tensor(init_filter).to(self.device)
                    self.Y = init_filter.mm(X) 
                    self.W = self.M.mm(init_filter)
                elif init == 'whiten':
                    init_filter = ortho_group.rvs(self.n)[:self.k]
                    init_filter = torch.Tensor(init_filter).to(self.device)
                    U_x, S_x, Vh_x = torch.svd(X.mm(X.t())/X.size(1))
                    init_filter = init_filter.mm(U_x.mm(torch.diag(1/torch.sqrt(S_x))).mm(U_x.t()))
                    self.Y = init_filter.mm(X) 
                    self.W = self.M.mm(init_filter)
                elif isinstance(init, Dict):
                    self.Y = init['Y']
                    self.W = init['W']
                    self.M = init['M']
                self.W = self.proj['W'](self.W)
                self.Y = self.proj['Y'](self.Y)
                self.M = self.proj['M'](self.M)
            
        eta_Y, eta_W, eta_M = self.eta_Y, self.eta_W, self.eta_M
        
        # solving the original problem, offline only
        if mode == 'YWM':
            errs_W = errs['W'] if errs else 1e-6
            errs_M = errs['M'] if errs else 1e-6

            if closed_form:
                if self.multi_inputs:
                    for i, nx in enumerate(self.n):
                        self.W[i] = closed_form['W'](self.Y, self.W[i], self.M, X[i]) 
                else:
                    self.W = closed_form['W'](self.Y, self.W, self.M, X)
                self.M = closed_form['M'](self.Y, self.W, self.M, X)
            else:
                if self.multi_inputs:
                    for i, nx in enumerate(self.n):
                        _step = 0
                        delta_W = self.corr(self.Y, X[i]) - self.dPhi(self.W[i], X[i])
                        while (delta_W * delta_W).sum() > errs_W and _step < step_limit:
                            _step += 1
                            self.W[i] = self.proj['W'](self.W[i] + eta_W * delta_W)
                            delta_W = self.corr(self.Y, X[i]) - self.dPhi(self.W[i], X[i])        
                else:
                    _step = 0
                    delta_W = self.corr(self.Y, X) - self.dPhi(self.W, X)
                    while (delta_W * delta_W).sum() > errs_W and _step < step_limit:
                        _step += 1
                        self.W = self.proj['W'](self.W + eta_W * delta_W)
                        delta_W = self.corr(self.Y, X) - self.dPhi(self.W, X)

                _step = 0
                delta_M = 0.5 * (self.corr(self.Y, self.Y) - self.dPsi(self.M, X))
                while (delta_M * delta_M).sum() > errs_M and _step < step_limit:
                    _step += 1
                    self.M = self.proj['M'](self.M + eta_M * delta_M)
                    delta_M = 0.5 * (self.corr(self.Y, self.Y) - self.dPsi(self.M, X))
                    
            if self.multi_inputs:
                delta_Y = - self.M.mm(self.Y)
                for i, nx in enumerate(self.n):
                    delta_Y += self.W[i].mm(X[i])
                delta_Y = delta_Y / self.Y.size(1)
            else:
                delta_Y = (self.W.mm(X) - self.M.mm(self.Y)) / X.size(1)
            self.Y = self.proj['Y'](self.Y + eta_Y * delta_Y)
        
        # solving the dual problems with NN, can be both online or offline
        elif mode=='NN':
            errs_Y = errs['Y'] if errs else 1e-6
            
            if closed_form:
                self.Y = closed_form['Y'](self.Y, self.W, self.M, X)
            else:
                _step = 0
                if self.multi_inputs:
                    delta_Y = - self.M.mm(self.Y)
                    for i, nx in enumerate(self.n):
                        delta_Y += self.W[i].mm(X[i])
                    delta_Y = delta_Y / self.Y.size(1)
                else:
                    delta_Y = (self.W.mm(X) - self.M.mm(self.Y)) / X.size(1)
                while (delta_Y * delta_Y).sum() > errs_Y and _step < step_limit:
                    _step += 1
                    self.Y = self.proj['Y'](self.Y + eta_Y * delta_Y)
                    if self.multi_inputs:
                        delta_Y = - self.M.mm(self.Y)
                        for i, nx in enumerate(self.n):
                            delta_Y += self.W[i].mm(X[i])
                        delta_Y = delta_Y / self.Y.size(1)
                    else:
                        delta_Y = (self.W.mm(X) - self.M.mm(self.Y)) / X.size(1)
            
            if self.multi_inputs:
                for i, nx in enumerate(self.n):
                    delta_W = self.corr(self.Y, X[i]) - self.dPhi(self.W[i], X[i])
                    self.W[i] = self.proj['W'](self.W[i] + eta_W * delta_W)
            else:
                delta_W = self.corr(self.Y, X) - self.dPhi(self.W, X)
                self.W = self.proj['W'](self.W + eta_W * delta_W)
            
            delta_M = 0.5 * (self.corr(self.Y, self.Y) - self.dPsi(self.M, X))
            self.M = self.proj['M'](self.M + eta_M * delta_M)
        
        # general projected graidient decent ascent, can be both online or offline 
        elif mode == 'GDA':
            if self.multi_inputs:
                delta_Y = - self.M.mm(self.Y)
                for i, nx in enumerate(self.n):
                    delta_Y += self.W[i].mm(X[i])
                    delta_W = self.corr(self.Y, X[i]) - self.dPhi(self.W, X[i])
                    
                    self.W[i] = self.proj['W'](self.W[i] + eta_W*delta_W)
                    
                delta_M = 0.5*(self.corr(self.Y, self.Y) - self.dPsi(self.M, X))
                
                self.M = self.proj['M'](self.M + eta_M*delta_M)
                self.Y = self.proj['Y'](self.Y + eta_Y*delta_Y)
            else:
                delta_Y = self.W.mm(X) - self.M.mm(self.Y)
                delta_W = self.corr(self.Y, X) - self.dPhi(self.W, X)
                delta_M = 0.5*(self.corr(self.Y, self.Y) - self.dPsi(self.M, X))

                self.M = self.proj['M'](self.M + eta_M*delta_M)
                self.W = self.proj['W'](self.W + eta_W*delta_W)
                self.Y = self.proj['Y'](self.Y + eta_Y*delta_Y)
    
    
    # evaluate objective function
    def objective(self, X):
        if self.multi_inputs:
            obj =  - 0.5 * (self.M * self.corr(self.Y, self.Y)).sum() + 0.5 * self.Psi(self.M, X)
            for i, nx in enumerate(self.n):
                obj += (self.W[i] * self.corr(self.Y, X[i])).sum() - self.Phi(self.W[i], X[i])
            return obj
        else:
            return (self.W * self.corr(self.Y, X)).sum() - 0.5 * (self.M * self.corr(self.Y, self.Y)).sum()\
                    - self.Phi(self.W, X) + 0.5 * self.Psi(self.M, X)