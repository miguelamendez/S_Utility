"""[World Model is the wrapper for the forward model used by the agent]
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
class World_Model_Wrapper():
	def __init__(self, output_dims,input_dims=512,batch_size=4,lr=.0003,chkpt_dir= directory,use_cuda=True):
		"""[summary]
		Args:
			input_dims ([type]): [tipicaly actions dim, assumend to be descrete] . 
			output_dims ([type]): [description] . Defaults to 512
			batch_size ([type]): [description] Default to 4
			lr (float, optional): [description]. Defaults to .0003.
			chkpt_dir ([type], optional): [description]. Defaults to directory.
		"""  
		super(World_Model_Wrapper, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, 'world_model')
		self.world_model =ForwardICMBaselineNetwork(output_dims)
		self.memory= ReplayMemoryWorldModel(batch_size,model_type)
		self.input_dims=input_dims

	def save_checkpoint(self):
		"""[save models]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		torch.save(self.world_model.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		"""[Load models]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		self.world_model.load_state_dict(torch.load(self.checkpoint_file))

	def forward_model_loss(self):
		"""[Not Implemented yet]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		return loss

	def compute_world_reward(self, state, next_state, action):
		"""[Compute Intrinsic Reward like in curiosity paper for world model. Implemented but not tested may be wrong]
		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		state = torch.FloatTensor(state).to(self.device)
		next_state = torch.FloatTensor(next_state).to(self.device)
		action = torch.LongTensor(action).to(self.device)
		#action_onehot = torch.FloatTensor(len(action), self.output_size).to(self.device)
		#action_onehot.zero_()
		#action_onehot.scatter_(1, action.view(len(action), -1), 1)
		real_next_state_feature, pred_next_state_feature, pred_action = self.world_model([state, next_state, action])
		intrinsic_reward = self.world_eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
		return intrinsic_reward.data.cpu().numpy()

	def pred_intrinsic_world_value(self):
		"""[Optional function - Not Implemented yet]"""
		return value
