"""
    Description: This library contains all the functions/architectures derived from the LSP and FSP models
    Please refer to each of the functions/classes for a full description of what they do.

    Functions (True are the implemented Functions):
        walshMatrix:
        SP_Matrix:
        SP_r_Matrix:
        frequencies_gen:
        GD_MSE_SP_step:
        MSELoss:

    Classes (True are the implemented classes):
        SP_numpy:True. Implementation of the signal perceptron using the "numpy" library (USED IN PAPER EXPERIMENTS)
        SP_r_numpy:True. Implementation of the real signal perceptron using the "numpy" library (USED IN PAPER EXPERIMENTS)
        SP_pytorch:False. Implementation of the signal perceptron using the "pytorch" library (Not functional as pytorch has problems to calculating complex gradients with respect of complex numbers)
        RSP_pytorch:True. Implementation of the real signal perceptron using the "pytorch" library (USED IN PAPER EXPERIMENTS)
        FSP_pytorch:True. Implementation of the real signal perceptron with learnable frequencies and fixed ammount of signals (USED IN THE PAPER EXPERIMENTS)
           """

#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("[archs][sp]:asp_baselines.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#print(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#print("Internal libraries:")
from ml.archs.sp import  baselines as sp

#External libraries
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
import math
#2d fourier signal perceptron (Shared frequencies)
class FSP_2D(nn.Module):
    def __init__(self,inout,parameters):
        k, heads_n, heads_m=inout
        n = parameters
        super(FSP_2d, self).__init__()
        self.m = n
        self.k = k
        self.heads_n= heads_n
        self.heads_m= heads_m
        self.freq = nn.Linear(k, n,bias=False)
        self.alphas =nn.Linear(n, heads_n*heads_m,bias=False)

    def forward(self, x):
        freq=self.freq(x)
        signals=torch.cos(freq)
        x=self.alphas(signals)
        x=torch.reshape(x, (len(x),self.heads_n, self.heads_m))
        return x

#Multivariate fourier signal perceptron (not shared frequencies)
class MultivarFSP(nn.Module):
    def __init__(self,inout,parameters):
        inputs,outputs=inout
        variables,signals=parameters
        super(MultivarFSP, self).__init__()
        self.var=variables
        self.outputs=outputs
        self.mfsp = nn.ModuleList([FSP_pytorch(signals,inputs,outputs) for i in range(variables)])
        
    def forward(self, x):
        mult_logits=[]
        for i in x:
            sing_logits=[]
            for n in range(0,self.var):
                sing_logits.append(self.mfsp[n](i))
            mult_logits.append(torch.stack(sing_logits))
        return torch.stack(mult_logits)

#Self-Attention Signal Perceptron not correctrly implemented
class Att_FSP(nn.Module):
    def __init__(self,inout,parameters):
        seq_leng,embed_dim= inout
        linear,sig_q,sig_k= parameters
        if linear:
            self.q=nn.Linear(k, n,bias=False)
            self.k=nn.Linear(k, n,bias=False)
        else:
            self.q=Att_FSP(sig_q,embed_dim,outputs)
            self.k=Att_FSP(sig_k,inputs,outputs)
        self.v=nn.Linear(k, n,bias=False)
        super(Att_FSP_pytorch, self).__init__()
        self.var=variables
        self.outputs=outputs
        self.mfsp = nn.ModuleList([FSP_pytorch(signals,inputs,outputs) for i in range(variables)])
        
    def forward(self, x):
        mult_logits=[]
        for i in x:
            sing_logits=[]
            for n in range(0,self.var):
                sing_logits.append(self.mfsp[n](i))
            mult_logits.append(torch.stack(sing_logits))
        return torch.stack(mult_logits)
