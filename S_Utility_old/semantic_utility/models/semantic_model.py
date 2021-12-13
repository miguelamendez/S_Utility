"""[Semantic Model Library si the library that contains the baselines for learning Symbolic representations of the environment and caculate the semantic utility ]
   There is to implementations of the semantic model:
   1.- When the semantic model is integrated on top of the forward model (that means the input would be the predictions of the forward model)
   2.- When the semantic model is integrated separately from the forward model (that means the input is the encoded state and action)
"""
import os
import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.categorical import Categorical

from semantic_utility.memory.replay_memory import *
from semantic_utility.models.nn_baselines import *
from semantic_utility.utils.constraint_utils import *

directory = os.getcwd()

#Wrapper Class for testing different semantic models
class Semantic_Model_Wrapper():
    def __init__(self,input_dims,output_dims,path,theory=[],batch_size=20,chkpt_dir= directory,use_cuda=True):
        """[summary]

        Args:
            input_dims   ([type]): [For deep weights: usually the encoded state + actions] . Defaults to
            output_dims ([type]): [For deep weights: the probability distribution over the literals] . Defaults to
            path ([type]): [description] . Inputs and outputs path
            arg4 ([type]): [description] . Defaults to
            """
        super(Semantic_Model_Wrapper, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'semantic_model')
        self.model =DeepWeightNetwork(input_dims,output_dims)
        self.const_processor = Constraint_Processor(theory,path)
        self.memory= ReplayMemorySemanticModel(batch_size)    
        #Model specific hyperparameters (will probably change)


    def save_checkpoint(self):
        """[summary]
        Args:
            arg1 ([type]): [description] . Defaults to
            arg2 ([type]): [description] . Defaults to
            arg3 ([type]): [description] . Defaults to
        """
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """[summary]
        
        Args:
            arg1 ([type]): [description] . Defaults to
            arg2 ([type]): [description] . Defaults to
            arg3 ([type]): [description] . Defaults to
        """
        self.model.load_state_dict(torch.load(self.checkpoint_file))


    def deepweights_loss(self,inputs,target):
        """[This is the loss function for deepweights. [Tested:False]]
        Args:
            inputs ([type]): [state_action vector] . Defaults to
            literals ([type]): [literals (obtained by processing some variables)] . Defaults to
        """
        y_pred=self.model(inputs)
        y_target =target
        loss = nn.MSELoss()
        loss = loss(y_pred,y_target)
        return loss
    
    def constraint_loss(self,inputs,constraint_sat_target):
        y_pred=self.model(inputs)
        const_pred=self.const_processor.wmc(y_pred)
        loss = MSE(const_pred,const_sat_target)
        return loss

    def compute_semantic_utility(self, inputs):
        """[ Not Implemented] 
        Args:
            arg1 ([type]): [description] . Defaults to
            arg2 ([type]): [description] . Defaults to
            arg3 ([type]): [description] . Defaults to
        """
        y_pred=self.model(inputs)
        const_pred=self.const_processor.wmc(y_pred)
        return const_pred.data.cpu().numpy()

    def pred_intrinsic_value(self):
        """[Optional function - Not Implemented yet 
        This should aproximate the semantic value wich isgiven the current policy and state what is the value function that we can achieve, 
        this will required to be aproximated by a neural network]"""
        return value
