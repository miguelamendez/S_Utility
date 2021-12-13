"""
    Description: This library contains all the functions/classes used for: Learning models
    The library is divided into: Objective functions 
    Please refer to each of the functions/classes for a full description of what they do.

    Functions ([first,second] "first bool" notes the implemented functions "second bool" notes the implemented documentation):
        name:[True,False] summary
        name:[True,False] su
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
print("obj_func.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# two directories above
#from dir.to.file import * #files for imports
#from dir.to.file import *

#External libraries
import numpy as np
import torch
import torch.nn.functional as F

class Obj_func():
    def __init__(self,func_list,weights):
        self.func_list=func_list
        self.weights=weights
        
    def forward(self,x):
        obj=[]
        for idx, values in enumerate(x):
            obj.append(self.func_list[idx](values))
        obj_tensor=torch.stack(obj)
        final_obj=self.weights*obj_tensor
        return final_obj.sum()
#Example:
#loss=Obj_func([lambda x: 2*x.sum(),lambda x:3*x.sum()],torch.ones(2))
#x=torch.ones(3)
#print(loss.forward([x,x]))

def nll(logits,dist,value):
    dist=dist_dic[dist_id](logits)
    loglike=dist.log_prob(value)
    return -loglike

def policy_gradient(logits,dist_id,value,advantage):
    neg_log_prob=nll(logits,dist_id,value)
    return neg_log_prob* advantage
    
def pg_is(logits,dist_id,value,advantage):
    neg_log_prob=nll(logits,dist_id,value)
    return neg_log_prob* advantag
    
def ppo(logits,dist_id,value,old_probs,advantage,policy_clip=.2):
        new_probs = dist.log_prob(actions)
        prob_ratio = new_probs.exp() / old_probs.exp()
        weighted_probs = advantage[batch] * prob_ratio
        weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,1+self.policy_clip)*advantage[batch]
