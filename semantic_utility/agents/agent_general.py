"""[Agent wrapper for learning for learning policies in RL Environments]
"""
import os
import torch
from collections import namedtuple
from semantic_utility.memory.replay_memory import *
from semantic_utility.models.semantic_model import *
from semantic_utility.models.policy_model import *
from semantic_utility.utils.dl_utils import *
from semantic_utility.utils.rl_utils import *
from semantic_utility.utils.analytics_utils import *
from semantic_utility.models.nn_baselines import *
import torch.nn as nn
from torchviz import make_dot

class Agent(object):
     def __init__(
               self,
               input_dims,
               output_dims,
               dtype,
               results,
               learn_epochs=4,
               learn_num_steps=32,
               gamma=0.99,
               gae_lambda=0.95,
               use_gae=True,
               policy_clip=0.1,
               lr=1e-4,
               policy_learning_rate=1e-4,
               world_learning_rate=1e-4,
               semantic_learning_rate=1e-4,
               entropy_coef=0.01,
               batch_size=5,
               world_eta=0.01,
               semantic_eta=0.01,
               world_model=False,
               inverse_model=False,
               semantic_model=True,
               image_model=False,
               semantic_utility_type ="semantic_reward",
               use_cuda=False):
          """[Generic Agent: This wrapper allows us to define an RL agent which can be of two categories (Model Based, Model Free), is model based we can choose the  world model. The code will change in future versions for making it more modular ]
          Args:
               input_dims ([array]): [The dimentions of the observations given by the environment] . 
               output_dims ([int]): [The dimentions of the action space (Current implementation only supports descrete actions)] . 
               dtype (str) : [The datatype of the input this is use to define the feature encoder]
               results (path): [The path were input and output folders is]
               learn_epochs([int]): [The number of epochs used by the training of the agent] . Defaults to 4
               learn_num_steps([]): [] . Defaults to 32
               ([]): [] . Defaults to .99
               ([]): [] . Defaults to .95
               ([]): [] . Defaults to True
               ([]): [] . Defaults to .1
               ([]): [] . Defaults to .0001
               ([]): [] . Defaults to .0001
               ([]): [] . Defaults to .0001
               ([]): [] . Defaults to .0001
               ([]): [] . Defaults to .01
               ([]): [] . Defaults to 128
               ([]): [] . Defaults to .01
               ([]): [] . Defaults to .01
               ([]): [] . Defaults to a2c
               ([]): [] . Defaults to curiosity
               ([]): [] . Defaults to dwmc
               semantic_utility_type ([str]): [How the semantic model will affect the policy rewards . Options --> semantic_reward , semantic_weight] . Defaults to semantic_reward
               ([]): [] . Defaults to True
               
          """
          """[Parameters]"""
          self.input_dims=input_dims
          self.output_dims=output_dims
          self.dtype = dtype
          self.learn_epochs=learn_epochs
          self.learn_num_steps=learn_num_steps
          self.gamma=gamma
          self.gae_lambda=gae_lambda
          self.policy_clip=policy_clip
          self.policy_learning_rate=policy_learning_rate
          self.world_learning_rate=world_learning_rate
          self.semantic_learning_rate=semantic_learning_rate
          self.entropy_coef=entropy_coef
          self.batch_size=batch_size
          self.world_eta=world_eta
          self.semantic_eta=semantic_eta
          self.use_world_model=world_model
          self.use_semantic_model=semantic_model
          self.use_image_model=image_model
          self.use_inverse_model = inverse_model
          self.semantic_utility_type=semantic_utility_type
          """[Models used by the agent:]"""
          if dtype == "Images":
               self.feature_encoder_model = CNNStateEncoderBaselineNetwork(input_dims)
          elif dtype == "Values":
               self.feature_encoder_model = LinearStateEncoderBaselineNetwork(input_dims)
          self.critic_model = LinearValueBaselineNetwork()
          self.policy= Policy_Model_Wrapper(output_dims)
          self.device = torch.device('cuda' if use_cuda else 'cpu')
          self.feature_encoder_model = self.feature_encoder_model.to(self.device)
          self.feature_encoder_model_param_list=list(self.feature_encoder_model.parameters())
          self.critic_model = self.critic_model.to(self.device)
          self.critic_model_param_list=list(self.critic_model.parameters())
          self.actor_model = self.policy.policy_model.to(self.device)
          self.actor_model_param_list=list(self.actor_model.parameters())
          if self.use_inverse_model:
               self.inverse_model = LinearInverseBaselineNetwork(output_dims)
               self.inverse_model = self.inverse_model.to(self.device)
               self.inverse_model_param_list = list(self.inverse_model.parameters())
          else:
               self.inverse_model_param_list = []
          if self.use_world_model:
               self.world = World_Model_Wrapper()
               self.forward_model = self.world.world_model.to(self.device)
               self.forward_model_param_list = list(self.foward_model.parameters())
          else:
               self.forward_model_param_list = []
          if self.use_image_model:
               self.image = Image_Model_Wrapper()
               self.image_model = self.image.image_model.to(self.device)
               self.image_model_param_list = list(self.image_model.parameters())
          else:
               self.image_model_param_list = []
          if self.use_semantic_model:
               self.semantic = Semantic_Model_Wrapper(
               input_dims = 512,output_dims=4,path=results)
               self.semantic_model = self.semantic.model.to(self.device)
               self.semantic_model_param_list = list(self.semantic_model.parameters())
               #self.memory= ReplayMemoryAgentSafety(batch_size)
          else:
               self.semantic_model_param_list = []
          self.memory= ReplayMemoryAgent(batch_size)
          #self.optimizer = optim.Adam(list(self.actor_model.parameters())+list(self.critic_model.parameters())+list(self.feature_encoder_model.parameters()),lr=lr)
          self.agent_optimizer = optim.Adam(self.actor_model_param_list+self.critic_model_param_list+self.feature_encoder_model_param_list+self.semantic_model_param_list,lr=0.0003)
          chkpt_dir=os.getcwd()
          self.checkpoint_file = os.path.join(chkpt_dir, 'agent_models')
##################################################################################################
     """[Agent functions are devided agent model functions,  loss functions , Advantage functions and  learning function. If not implemented it will have a comment inside]"""
     def  get_agent_id(self):
          """[Function that generates a name depending on the models used by the agent]
          """
          name_id = "actor_critic_model"
          if self.use_world_model:
               name_id =name_id+"_world_model"
          if self.use_semantic_model:
               name_id =name_id+"_semantic_model"
               name_id =name_id+"_"+self.semantic_utility_type
          if self.use_image_model:
               name_id =name_id+"_image_model"
          return name_id

     def encode_state(self,state):
          """[Function for encoding the state using the feature encoder model]
          Args:
               state ([ Gym type]): [state from the environment] . 
          """
          state = torch.Tensor(state).to(self.device)
          state = state.float()
          
          enc_state = self.feature_encoder_model(state)
          return enc_state

     def choose_action(self,state):
          """[Function for choosing an Action using the actor model from Policy_Model wrapper]
          Args:
               state ([ Gym type]): [state extracted from the environment] . 
          """
          #if self.dtype == "Images":
          #     state = image_preprocess(state)
          enc_state = self.encode_state(state)
          action_dist = self.actor_model(enc_state)
          dist = Categorical(action_dist)
          action = dist.sample()
          #action_log_prob = dist.log_prob(action)
          action_log_prob = torch.squeeze(dist.log_prob(action)).item()
          action = torch.squeeze(action).item()
          return action , action_log_prob

     def pred_action(self,state,next_state):
          """[Function for predicting the action performed by the policy using the inverse  model ]
          Args:
               state ([ Gym type]): [state extracted from the environment] . 
               next_state ([ Gym type]): [ next state extracted from the environment] . 
          """
          enc_state = self.encode_state(state)
          enc_next_state = self.encode_state(next_state)
          pred_action = self.inverse_model([enc_state,enc_next_state])
          return pred_action

     def pred_state_value(self,state):
          """[Function for predicting the action performed by the policy using the inverse  model ]
          Args:
               state ([ Gym type]): [state extracted from the environment] . 
               next_state ([ Gym type]): [ next state extracted from the environment] . 
          """
          enc_state = self.encode_state(state)
          value = self.critic_model(enc_state)
          value = torch.squeeze(value).item()
          return value

     def remember(self,episode,state,next_state,action,action_log_prob, reward,value, done):
          self.memory.store_memory( episode,state,next_state,action,action_log_prob, reward,value, done)
#     def remember_su(self,episode,state,next_state,action,action_log_prob, reward,value,literal_actual_val,constraint_sat, done):
#          self.memory.store_memory( episode,state,next_state,action,action_log_prob, reward,value,literal_actual_val,constraint_sat,done)
##################################################################################################
#Loss functions

     def critic_model_loss(self,enc_state,extrinsic_advantage,extrinsic_value):
          """[Loss function for the critic model ]
          Args:
               arg1 ([type]): [description] . Defaults to
               arg2 ([type]): [description] . Defaults to
               arg3 ([type]): [description] . Defaults to
          """
          critic_value = self.critic_model(enc_state)
          critic_value = torch.squeeze(critic_value)
          returns = extrinsic_advantage + extrinsic_value
          critic_loss = (returns-critic_value)**2
          loss = critic_loss.mean()
          return loss

     def inverse_model_loss(self,state,next_state,action):
          """[Loss function for the inverse model not tested]
          Args:
               arg1 ([type]): [description] . Defaults to
               arg2 ([type]): [description] . Defaults to
               arg3 ([type]): [description] . Defaults to
          """
          pred_action=self.pred_action(self,state,next_state)
          loss=nn.CrossEntropyLoss(pred_action, action)
          return loss


##################################################################################################
#Advantage functions
     def total_advantage(self):
          """[Adds all the advantages if more than one advantage is used not Implemented yet]
          Args:
               arg1 ([type]): [description] . Defaults to
               arg2 ([type]): [description] . Defaults to
               arg3 ([type]): [description] . Defaults to
          """
          return advantage

     def compute_extrinsic_advantage(self,values,rewards,dones):
          """[Advantage function calculated using the rewards given by the environment and the value function given by the critic model]
          Args:
               reward ([type]): [reward batch extracted from the environment] . 
               arg2 ([type]): [value batch calculated by the critic model] . 
               arg3 ([type]): [description] . Defaults to
          """
          advantage = np.zeros(len(rewards), dtype=np.float32)
          for t in range(len(rewards)-1):
               discount = 1
               a_t = 0 
               for k in range(t, len(rewards)-1):
                         a_t += discount*(rewards[k] + self.gamma*values[k+1]*\
                              (1-int(dones[k])) - values[k])
                         discount *= self.gamma*self.gae_lambda
               advantage[t] = a_t
          return torch.tensor(advantage).to(self.device)

     def intrinsic_world_model_andvantage(self,rewards,values,dones):
          """[Advantage function for ICM not implemented yet]
          Args:
               arg1 ([type]): [description] . Defaults to
               arg2 ([type]): [description] . Defaults to
               arg3 ([type]): [description] . Defaults to
          """
          advantage = np.zeros(len(rewards), dtype=np.float32)
          for t in range(len(rewards)-1):
               discount = 1
               a_t = 0 
               for k in range(t, len(rewards)-1):
                         a_t += discount*(rewards[k] + self.gamma*values[k+1]*\
                              (1-int(dones[k])) - values[k])
                         discount *= self.gamma*self.gae_lambda
               advantage[t] = a_t
          return advantage

     def advantage_semantic_model(self,rewards,values,dones):
          """[Advantage function for Semantic Utility when is used as a reward not Implemented yet]
          Args:
               arg1 ([type]): [description] . Defaults to
               arg2 ([type]): [description] . Defaults to
               arg3 ([type]): [description] . Defaults to
          """
          return advantage.detach().numpy()

#Agent Learning
##################################################################################################
     def learn_model(self):
          """[Learning function for the agent working  but missing some parts when model based or semantic utility is used ( finish Implementation only if you already understood the whole code and fill up the missing parts)]
          """
          for _ in range(self.learn_epochs):
               episode_arr,state_arr, next_state_arr,action_arr, old_probs_arr,reward_arr,values_arr, dones_arr, batches = self.memory.generate_batches()
               advantage= self.compute_extrinsic_advantage(values_arr,reward_arr, dones_arr)
               values = torch.tensor(values_arr).to(self.device)
               #Second Inner loop samples using batch Indexes
               for batch in batches:
                    states = state_arr[batch]
                    enc_states= self.encode_state(states)
                    old_probs = torch.tensor(old_probs_arr[batch]).to(self.device)
                    actions = torch.tensor(action_arr[batch]).to(self.device)
                    #Actor Loss
                    actor_loss = self.policy.ppo_actor_loss(enc_states,actions,old_probs,advantage[batch])
                    #----------------------------------------------------------------------------------
                    #Critic Loss
                    critic_model_loss = self.critic_model_loss(enc_states,advantage[batch],values[batch])
                    #---------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
                    # Inverse Model Loss
                    if self.use_inverse_model:
                         inverse_model_loss=self.inverse_model_loss(state,next_state,action)
                    else:
                         inverse_model_loss=0
# --------------------------------------------------------------------------------
                    #  Semantic Model Loss
                    if self.use_semantic_model:
                         #semantic_model_loss = self.semantic.deepweights_loss()
                         semantic_model_loss = 0
                    else:
                         semantic_model_loss = 0
# --------------------------------------------------------------------------------
                    # World Model Loss
                    if self.use_world_model:
                         forward_model_loss = self.world.forward_model_loss()
                    else:
                         forward_model_loss = 0
                    # ---------------------------------------------------------------------------------
                    # Image Model Loss
                    if self.use_image_model:
                         image_model_loss = self.image.image_model_loss()
                    else:
                         image_model_loss = 0
                    #----------------------------------------------------------------------------------
                    #Total Loss
                    loss = actor_loss+0.5*critic_model_loss+semantic_model_loss
                    self.agent_optimizer.zero_grad()
                    #address = "gradients/"+self.get_agent_id()
                    #make_dot(loss).render(address, format="png")
                    loss.backward()
                    self.agent_optimizer.step()
          self.memory.clear_memory()
