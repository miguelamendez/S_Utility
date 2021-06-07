"""[Image Model is the wraper for the image used by the model IGNORE FOR THE SEMANTIC UTILITY implementation]
"""
import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from semantic_utility.memory.replay_memory import *
from semantic_utility.models.nn_baselines import *
directory = os.getcwd()


#Wrapper Class for testing different world models
class Image_Model_Wrapper():
	def __init__(self, output_dims,input_dims=512,batch_size=4,lr=.0003,chkpt_dir= directory,use_cuda=True):
		"""[Image Model is a wrapper to create a self learning agent this file is not used for the Semantic model PLEASE IGNORE but not delete]
		Args:
			input_dims ([type]): [tipicaly actions dim, assumend to be descrete] . 
			output_dims ([type]): [description] . Defaults to 512
			batch_size ([type]): [description] Default to 4
			lr (float, optional): [description]. Defaults to .0003.
			chkpt_dir ([type], optional): [description]. Defaults to directory.
		"""  
		super(Image_Model_Wrapper, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, 'image_model')
		self.image_model =ImageBaselineNetwork(output_dims)
		self.memory= ReplayMemoryImageModel(batch_size)
		self.input_dims=input_dims

	def save_checkpoint(self):
		"""[save models]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		"""[Load models]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		self.load_state_dict(torch.load(self.checkpoint_file))

	def image_model_loss(self):
		"""[Not Implemented yet]

		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		return loss

	def compute_intrinsic_image_reward(self, literals,state, next_state, action):
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
