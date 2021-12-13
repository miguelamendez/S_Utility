"""
    Description: This library contains all the functions/classes used for: probabilistic reasoning and Inference.
    The library is divided into: Probabilistic functions , Information Theory functions , other Statistical Analisys functions
    Please refer to each of the functions/classes for a full description of what they do.

    Functions ([first,second] "first bool" notes the implemented functions "second bool" notes the implemented documentation):
        probs_E:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary


    Classes ([first,second] "first bool" notes the implemented classes "second bool" notes the implemented documentation):
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("probs_func.py:",full_path)
sys.path.append(os.path.dirname(full_path))# one directory above
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# two directories above
#from dir.to.file import * #files for imports
#from dir.to.file import *

#External libraries
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import *
from torch.distributions import Categorical, Distribution
from typing import List

#Probabilistic functions-------------------------------------------------------------------------------------------------

def probs_E(p_x,x,func=[]):
    
    
    if bool(func):
        a=torch.matmul(func(x), p_x)
    else:
        a=torch.matmul(x, p_x)
        #print(a.sum())
    #Examples:
    #Single variable
    #x=torch.tensor([[1.],[2.],[3.]],requires_grad=True)
    #p_x=torch.tensor([[.25],[.5],[.25]],requires_grad=True)
    #x=torch.transpose(x, 0, 1)
    #print(x,p_x)
    #with function:
    #def g(x):
    #    return x**2
    #print(exp_value(x,p_x,g))
    #without function
    #print(exp_value(x,p_x))
    #Multivariable:
    #xy=torch.tensor([[0.,0.],[0.,1.],[1.,1.]],requires_grad=True)
    #p_xy=torch.tensor([[.25,.25,.25],[.25,.25,.25],[.25,.25,.25]],requires_grad=True)
    #xy=torch.transpose(xy, 0, 1)
    #print(exp_value(xy,p_xy))
    return a.sum()



def probs_var(x,p_x):
    g=(x-exp_value(x,p_x))**2
    #Example
    #x=torch.tensor([1,2,3])
    #p_x=torch.tensor([.25,.5,.25])
    #print(variance(x,p_x))
    return exp_value(g,p_x)
    
def probs_var_lambda(x,p_x):
    #Example
    #x=torch.tensor([1,2,3])
    #p_x=torch.tensor([.25,.5,.25])
    #print(variance_lambda(x,p_x))
    return exp_value(x,p_x,lambda x : (x-exp_value(x,p_x))**2)

def probs_corr_coef(x,p_x):
    x=torch.transpose(x, 0, 1)
    for i in x:
        g=g+(i-exp_value(i,p_x))**2
    return exp_value(g,p_x)

#Information Theory functions ----------------------------------------------------------------------------------------------------
def it_I(p):
    return -torch.log(p)

def it_KL():
   return 
def it_CE(p,q):
    return
def it_E(p,q):
    return
#Statistical functions--------------------------------------------------------------------------------------------------------------

class MultiCategorical(Distribution):
    arg_constraints = {}
    def __init__(self, logits):
        super().__init__()
        self.dists = self.build_dist(logits)

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)
    
    def build_dist(self,logits):
        #Logits first Dim=0 is batch size, Dim=1 is number of variables Dim=2 is number of categories
        nvec=logits.size()
        dists = []
        for n in range(0,nvec[1]):
            dists.append(Categorical(logits=logits[:,n]))
        return dists


#ML functions
def stg(dist,logits,num_class=2,dim=0,binary=False):
    #Multivariate-Bernoulli Dist binary =True
    #Univariate-Categorical Dist dim = 1
    #Multivariate-Categorical Dist dim = 2
    #print("stg:")
    if binary:
        m=nn.Sigmoid()
        x = dist.sample()
        #print(x)
        sample =x + m(logits) - m(logits).detach()
        return sample
    else:
        m=nn.Softmax(dim=1)
        x = dist.sample()
        sample_grad =F.one_hot(x,num_class) + m(logits) - m(logits).detach()
        return x, sample_grad
