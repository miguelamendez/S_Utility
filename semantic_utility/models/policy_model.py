"""[Policy Model is  the  wrapper for the actor used by the Agent ]
"""
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from semantic_utility.memory.replay_memory import *
from semantic_utility.models.nn_baselines import *
directory = os.getcwd()

#Wrapper Class for testing different policy models
class Policy_Model_Wrapper():
	def __init__(self, input_dims,output_dims,batch_size=5,lr=.0003,chkpt_dir= directory,use_gae=True,use_cuda=True):
		"""[This class contains the actor and its loss function which currently is assumed to be ppo for actor]

		Args:
			input_dims ([type]): [tipicaly  encoded state dimention] Default to 512
			output_dims ([type]): [the number of descrete actions]
			batch_size ([type]): [description] Default to 4
			lr (float, optional): [description]. Defaults to .0003.
			chkpt_dir ([type], optional): [description]. Defaults to directory.
		"""     
		super(Policy_Model_Wrapper, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, 'policy_model')
		self.memory= ReplayMemoryPolicyModel(batch_size)
		self.policy_model = LinearActorBaselineNetwork(input_dims,output_dims)
		
		#Objective Functions Hyperparameters:
		
		
		self.action_dim = output_dims
		self.state_dim = input_dims
		self.batch_size = batch_size
		self.policy_gamma =0.99
		self.policy_clip =0.26
		self.n_epochs =4
		self.gae_lambda =0.95
		self.l_rate =0.0003

	def save_checkpoint(self):
		"""[summary]

		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		torch.save(self.policy_model.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		"""[summary]

		Args:
			arg1 ([type]): [description] . Defaults to
			arg2 ([type]): [description] . Defaults to
			arg3 ([type]): [description] . Defaults to
		"""
		self.policy_model.load_state_dict(torch.load(self.checkpoint_file))

#Learning different models (may be change to a differerent file)BEGIN--------------------------------------------------------------
	def ppo_actor_loss(self,enc_state,action,old_action_log_prob,advantage):
		"""[loss function for ONLY the Actor ]

		Args:
		arg ([type]): [description] . Defaults to
		arg ([type]): [description] . Defaults to
		arg ([type]): [description] . Defaults to
		"""  
		dist = self.policy_model(enc_state)
		dist = Categorical(dist)
		entropy = dist.entropy().mean()
		new_action_log_prob = dist.log_prob(action)
		prob_ratio = new_action_log_prob.exp() / old_action_log_prob.exp()
		weighted_probs = advantage* prob_ratio
		weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,1+self.policy_clip)*advantage
		actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
		loss =  (actor_loss  -0.0001*entropy) 
		return loss
