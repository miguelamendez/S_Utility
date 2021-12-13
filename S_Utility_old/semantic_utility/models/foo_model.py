"""[Foo Model is a template for creating new learning models that can be added to the agent IGNORE FOR THE SEMANTIC UTILITY implementation]
"""
#Some basic imports
import os
import numpy as np
import math

#torch imports
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.categorical import Categorical

#semantic utility imports
from nn_baselines import *
from replay_memory import *

directory = os.getcwd()


#Wrapper Class for testing different world models
class Model_Wrapper():
	def __init__(self, output_dims,nn_baseline,name,memory,chkpt_dir= directory,use_cuda=True):
		"""[Image Model is a wrapper to create a self learning agent this file is not used for the Semantic model PLEASE IGNORE but not delete]
		Args:
			input_dims ([type]): [tipicaly actions dim, assumend to be descrete] . 
			output_dims ([type]): [description] . Defaults to 512
			batch_size ([type]): [description] Default to 4
			lr (float, optional): [description]. Defaults to .0003.
			chkpt_dir ([type], optional): [description]. Defaults to directory.
		"""  
		super(Foo_Model_Wrapper, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, name)
		self.model =nn_baseline
		self.memory= memory
		self.input_dims=input_dims

	def save_checkpoint(self):
		"""[save models]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		torch.save(self.model.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		"""[Load models]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		self.model.load_state_dict(torch.load(self.checkpoint_file))

	def model_loss(self):
		"""[Not Implemented yet]

		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		return loss

	def get_intrinsic_reward(self, inputs,target):
		"""[Optional IGNORE FOR THE SEMANTIC UTILITY PAPER not Implemented]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		return intrinsic_image_reward.data.cpu().numpy()

	def pred_intrinsic_image_value(self):
		"""[Optional function - Not Implemented yet]"""
		return value
