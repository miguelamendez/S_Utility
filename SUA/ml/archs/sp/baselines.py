"""
    Description: This library contains all the base functions used for defining the signal perceptron
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
print("[archs][sp]:baselines.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#print(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#
from utils.matrices import *


#External libraries
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
import math

def SP_Matrix(m,k):
    aix=np.zeros([k]); #Array of indexes (to order them)
    aiw=np.zeros([k]); #Array of indexes (to order them)
    ni=m**k   #Number of Iterations
    n=k  #No. of variables
    nn=m**n #|m^k| domain space
    nnn=m**nn #|Delta|=|m^m^k| function space
    #Matrix
    A=np.zeros([nn,nn],dtype=complex) 
    divfrec=m-1
    i=0; j=0
    v=0; 
    for xi in range(0,ni,1):
        kx=xi;
        for xj in range(0,k,1): 
            aix[xj]= int ( kx % m ); 
            kx=int(kx/m); 
            #print("aix=",aix)
            j=0;
        #First Inner nested loop that generates all combinations of w for a signal
        for wi in range(0,ni,1):
            kw=wi;
            for wj in range(0,k,1): #Generamos los índices
                aiw[wj]= int ( kw % m ) ; #Lo metemos en array 
                kw=int(kw/m); #siguientes índices
                #print(i,j,A[i,j],"|",end='')
            exponente=0
            #Seconf Inner loop that  multiplies and sums
            for ii in range(0,k,1):
                exponente=exponente + aix[ii]*aiw[ii]
                exponente=int(exponente)
            #print("exponente=",exponente)
            exponente=1j*np.pi*exponente/divfrec
            #print(exponente)
            #print(np.exp(exponente))
            A[i,j]=np.exp(exponente)
            #print(A[i,j])
            j=j+1
            #print("aiw=",aiw,"j=",j)
            #for aj in range(0,nc,1):
            #	print(i,j,A[i,j],"|",end='')
            #	print()
        i=i+1
        
    return A 

def RSP_Matrix(m,k):
    aix=np.zeros([k]); #Array of indexes (to order them)
    aiw=np.zeros([k]); #Array of indexes (to order them)
    ni=m**k   #Number of Iterations
    n=k  #No. of variables
    nn=m**n #|m^k| domain space
    nnn=m**nn #|Delta|=|m^m^k| function space
    # Matrix
    A=np.zeros([nn,nn],dtype=np.float32) 
    divfrec=m-1
    i=0; j=0
    v=0; 
    for xi in range(0,ni,1):
        kx=xi;
        for xj in range(0,k,1): 
            aix[xj]= int ( kx % m ); 
            kx=int(kx/m); 
            #print("aix=",aix)
            j=0;
        #First Inner nested loop that generates all combinations of w for a signal
        for wi in range(0,ni,1):
            kw=wi;
            for wj in range(0,k,1): #Generamos los índices
                aiw[wj]= int ( kw % m ) ; #Lo metemos en array 
                kw=int(kw/m); #siguientes índices
                #print(i,j,A[i,j],"|",end='')
            exponente=0
            #Seconf Inner loop that  multiplies and sums
            for ii in range(0,k,1):
                exponente=exponente + aix[ii]*aiw[ii]
                exponente=int(exponente)
            #print("exponente=",exponente)
            exponente=np.pi*exponente/divfrec
            #print(exponente)
            #print(np.exp(exponente))
            A[i,j]=np.cos(exponente)
            #print(A[i,j])
            j=j+1
            #print("aiw=",aiw,"j=",j)
            #for aj in range(0,nc,1):
            #	print(i,j,A[i,j],"|",end='')
            #	print()
        i=i+1
    return A


def GD_MSE_SP_step(Y, X, model,lr):
    N=len(X)
    #Calculate the gradient
    pred ,m_exp= model.forward(X)
    #print("pred",pred,"real",Y)
    gradient= -2/N*np.dot((Y-pred),m_exp)
    #print("grad",gradient)
    #Update parameters
    model.alphas = model.alphas - lr * gradient


    
#Signal Perceptron Classes:

class SP_numpy(object):
    def __init__(self,m,k,heads=1):
        self.m=m
        self.freq=frequencies_gen(m,k)
        self.init_alphas=.5 * np.random.randn(heads, m**k)
        self.alphas=self.init_alphas.copy()
    #print("frecuency matrix",arrw.shape)
    def forward(self,x):
        #print("x",x.shape)
        x = np.transpose(x)
        #print("x trans",x.shape)
        exp=np.dot(self.freq,x)
        #print("exponent",exp.shape)
        o_sp=np.exp((1j*np.pi/(self.m-1))*exp)
        #print("after exponential",o_sp)
        #print("theta vector",theta.shape)
        y_sp=np.dot(self.alphas,o_sp)
        #print("result",y_sp)
        return y_sp , o_sp
    def count(self):
        return self.alphas.size
    def reset_params(self):
        self.alphas=self.init_alphas
    def load_params(self,alphas):
        self.alphas=alphas

class RSP_numpy(object):
    def __init__(self,m,k,heads=1):
        self.m=m
        self.freq=freq_gen_sp(m,k)
        self.init_alphas=.5 * np.random.randn(heads, m**k)
        self.alphas=self.init_alphas.copy()
    #print("frecuency matrix",arrw.shape)
    def forward(self,x):
        #print("x",x.shape)
        x = np.transpose(x)
        #print("x trans",x.shape)
        exp=np.dot(self.freq,x)
        #print("exponent",exp.shape)
        o_sp=np.cos((np.pi/(self.m-1))*exp)
        #print("after exponential",o_sp)
        #print("theta vector",theta.shape)
        y_sp=np.dot(self.alphas,o_sp)
        #print("result",y_sp)
        return y_sp , o_sp
    def count(self):
        return self.alphas.size
    def reset_params(self):
        self.alphas=self.init_alphas
    def load_params(self,alphas):
        self.alphas=alphas

class SP_pytorch(nn.Module):
    def __init__(self,inout, parameters):
        k, heads = inout
        m = parameters 
        super(SP_pytorch, self).__init__()
        self.m = m
        params = torch.from_numpy(freq_gen_sp(m,k))
        self.freq = nn.Linear(k, m**k,bias=False)
        self.freq.weight = torch.nn.Parameter(params)
        for param in self.freq.parameters():
            param.requires_grad = False
        self.alphas_real= nn.Linear(m**k, heads,bias=False)
        self.alphas_imag= nn.Linear(m**k, heads,bias=False)
        
    def forward(self, x):
        freq=self.freq(x)
        signals_real=torch.cos((torch.tensor(math.pi)/(self.m-1))*freq)
        signals_imag=torch.sin((torch.tensor(math.pi)/(self.m-1))*freq)
        real=self.alphas_real(signals_real)-self.alphas_imag(signals_imag)
        imag=self.alphas_real(signals_imag)+self.alphas_imag(signals_real)
        x=torch.stack((real,imag),-1)
        z=torch.view_as_complex(x)
        return z , x

#Approximate signal perceptron deprecates should use Laplace signal perceptron
class ASP_pytorch(nn.Module):
    def __init__(self,inout,parameters):
        k, heads=inout
        n = parameters
        super(ASP_pytorch, self).__init__()
        self.m = n
        self.k = k
        self.freq = nn.Linear(k, n,bias=False)
        self.alphas_real =nn.Linear(n, heads,bias=False)
        self.alphas_imag =nn.Linear(n, heads,bias=False)

    def forward(self, x):
        freq=self.freq(x)
        signals_real=torch.cos(freq)
        signals_imag=torch.sin(freq)
        real=self.alphas_real(signals_real)-self.alphas_imag(signals_imag)
        imag=self.alphas_real(signals_imag)+self.alphas_imag(signals_real)
        x=torch.stack((real,imag),-1)
        z=torch.view_as_complex(x)
        return z , x
#Example
#m=ASP_pytorch([8,4,1])
#x=torch.tensor([0.,0.,1.,0.])
#print(m(x).size())

#Laplace signal perceptron not functional :forward function not well defined
class LSP_pytorch(nn.Module):
    def __init__(self,inout,parameters):
        k,heads=inout
        n = parameters
        super(FSP_pytorch, self).__init__()
        self.m = n
        self.k = k
        self.freq_real = nn.Linear(k, n,bias=False)
        self.freq_imag = nn.Linear(k, n,bias=False)
        self.alphas_real =nn.Linear(n, heads,bias=False)
        self.alphas_imag =nn.Linear(n, heads,bias=False)

    def forward(self, x):
        signals_real=torch.cos(freq_real)
        signals_imag=torch.sin(freq_imag)
        real=self.alphas_real(signals_real)-self.alphas_imag(signals_imag)
        imag=self.alphas_real(signals_imag)+self.alphas_imag(signals_real)
        x=torch.stack((real,imag),-1)
        z=torch.view_as_complex(x)
        return x

#Real signal perceptron is the real version of the signal perceptron defined in the paper
class RSP_pytorch(nn.Module):
    def __init__(self,inout,parameters):
        m, k, heads = inputs
        super(RSP_pytorch, self).__init__()
        self.m = m
        params = torch.from_numpy(freq_gen_sp(m,k))
        self.freq = nn.Linear(k, m**k,bias=False)
        self.freq.weight = torch.nn.Parameter(params)
        for param in self.freq.parameters():
            param.requires_grad = False
        self.alphas= nn.Linear(m**k, heads,bias=False)

    def forward(self, x):
        freq=self.freq(x)
        signals=torch.cos((torch.tensor(math.pi)/(self.m-1))*freq)
        x=self.alphas(signals)
        return x

#Fourier signal perceptron
class FSP_pytorch(nn.Module):
    def __init__(self,inout, parameters):
        k,heads=inout
        n=parameters
        super(FSP_pytorch, self).__init__()
        self.m = n
        self.k = k
        self.freq = nn.Linear(k, n,bias=False)
        self.alphas =nn.Linear(n, heads,bias=False)

    def forward(self, x):
        freq=self.freq(x)
        signals=torch.cos(freq)
        x=self.alphas(signals)
        return x


