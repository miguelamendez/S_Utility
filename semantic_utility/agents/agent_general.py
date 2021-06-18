"""[Agent wrapper for learning for learning policies in RL Environments]
"""
import os
import torch
import torch.nn as nn

from collections import namedtuple
from semantic_utility.memory.replay_memory import *
from semantic_utility.models.semantic_model import *
from semantic_utility.models.policy_model import *
from semantic_utility.utils.dl_utils import *
from semantic_utility.utils.rl_utils import *
from semantic_utility.utils.analytics_utils import *
from semantic_utility.models.nn_baselines import *
from torchviz import make_dot


class Agent(object):
     def __init__(
               self,
               input_dims,
               output_dims,
               dtype,
               path,
               constraint=[],
               num_literals=-1,
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
          """[Generic Agent: This wrapper allows us to define an RL agent which can be of two categories (Model Based, Model Free).
                If it is model based we can choose the  world model. The code will change in future versions to make it more modular]

          Args:
              input_dims (list/int): [description]
              output_dims (int): [description]
              dtype (str): [description]
              path (str): [description]
              constraint (list, optional): [description]. Defaults to [].
              num_literals (int, optional): [description]. Defaults to -1.
              learn_epochs (int, optional): [description]. Defaults to 4.
              learn_num_steps (int, optional): [description]. Defaults to 32.
              gamma (float, optional): [description]. Defaults to 0.99.
              gae_lambda (float, optional): [description]. Defaults to 0.95.
              use_gae (bool, optional): [description]. Defaults to True.
              policy_clip (float, optional): [description]. Defaults to 0.1.
              lr (float, optional): [description]. Defaults to 1e-4.
              policy_learning_rate (float, optional): [description]. Defaults to 1e-4.
              world_learning_rate (float, optional): [description]. Defaults to 1e-4.
              semantic_learning_rate (float, optional): [description]. Defaults to 1e-4.
              entropy_coef (float, optional): [description]. Defaults to 0.01.
              batch_size (int, optional): [description]. Defaults to 5.
              world_eta (float, optional): [description]. Defaults to 0.01.
              semantic_eta (float, optional): [description]. Defaults to 0.01.
              world_model (bool, optional): [description]. Defaults to False.
              inverse_model (bool, optional): [description]. Defaults to False.
              semantic_model (bool, optional): [description]. Defaults to True.
              image_model (bool, optional): [description]. Defaults to False.
              semantic_utility_type (str, optional): 
                    How the semantic model will affect the policy rewards. Options --> semantic_reward, semantic_weight. Defaults to "semantic_reward".
              use_cuda (bool, optional): [description]. Defaults to False.
          """
          
          # [Parameters]
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

          # Device definition
          self.device = torch.device('cuda' if use_cuda else 'cpu')
          
          # Models used by the agent:
          enc_state_dims = 512
          if dtype == "Images":
               self.feature_encoder_model = CNNStateEncoderBaselineNetwork(input_dims, enc_state_dims)
          elif dtype == "Values":
               self.feature_encoder_model = LinearStateEncoderBaselineNetwork(input_dims, enc_state_dims)
          
          # Feature encoder model
          self.feature_encoder_model = self.feature_encoder_model.to(self.device)
          self.feature_encoder_model_param_list=list(self.feature_encoder_model.parameters())
          
          # Critic model
          self.critic_model = LinearValueBaselineNetwork(enc_state_dims)
          self.critic_model = self.critic_model.to(self.device)
          self.critic_model_param_list=list(self.critic_model.parameters())
          
          # Actor model
          self.policy= Policy_Model_Wrapper(input_dims=enc_state_dims, output_dims=output_dims)
          self.actor_model = self.policy.policy_model.to(self.device)
          self.actor_model_param_list=list(self.actor_model.parameters())
          
          # Inverse model
          if self.use_inverse_model:
               self.inverse_model = LinearInverseBaselineNetwork(enc_state_dims * 2, output_dims)
               self.inverse_model = self.inverse_model.to(self.device)
               self.inverse_model_param_list = list(self.inverse_model.parameters())
          else:
               self.inverse_model_param_list = []

          # World model
          if self.use_world_model:
               self.world = World_Model_Wrapper([enc_state_dims,output_dims])
               self.forward_model = self.world.world_model.to(self.device)
               self.forward_model_param_list = list(self.foward_model.parameters())
          else:
               self.forward_model_param_list = []
          
          # Image model
          if self.use_image_model:
               self.image = Image_Model_Wrapper()
               self.image_model = self.image.image_model.to(self.device)
               self.image_model_param_list = list(self.image_model.parameters())
          else:
               self.image_model_param_list = []
          
          # Semantic model
          if self.use_semantic_model:
               self.semantic = Semantic_Model_Wrapper(
               input_dims = [enc_state_dims,output_dims],output_dims=num_literals,path=path,theory=constraint)
               self.semantic_model = self.semantic.model.to(self.device)
               self.semantic_model_param_list = list(self.semantic_model.parameters())
          else:
               self.semantic_model_param_list = []
          
          # Memory
          self.memory= ReplayMemoryAgentSemantic(batch_size)
          
          # Optimizers and checkpoint files
          #self.optimizer = optim.Adam(list(self.actor_model.parameters())+list(self.critic_model.parameters())+list(self.feature_encoder_model.parameters()),lr=lr)
          self.agent_optimizer = optim.Adam(self.actor_model_param_list+self.critic_model_param_list+self.feature_encoder_model_param_list+self.semantic_model_param_list,lr=0.0003)
          chkpt_dir=os.getcwd()
          self.checkpoint_file = os.path.join(chkpt_dir, 'agent_models')
##################################################################################################


     # Agent functions are devided agent model functions, loss functions, advantage functions and learning function. If not implemented it will have a comment inside.
     def  get_agent_id(self):
          """Function that generates a name depending on the models used by the agent.

          Returns:
              str: A string represantation of the model used by the agent
          """
          name_id = "actor_critic_model"
          
          if self.use_world_model:
               name_id = name_id + "_world_model"
          if self.use_semantic_model:
               name_id = name_id + "_semantic_model" + "_" + self.semantic_utility_type
          if self.use_image_model:
               name_id = name_id + "_image_model"
          
          return name_id


     def encode_state(self, state):
          """Encodes a given state using the feature encoder model

          Args:
              state ([type]): The state of the environment

          Returns:
              [type]: A encoded representation of the specified state of the environment
          """
          state = torch.Tensor(state).to(self.device)
          state = state.float()

          enc_state = self.feature_encoder_model(state)
          return enc_state


     def choose_action(self, state):
          """ Chooses an action using the actor model from Policy_Model wrapper

          Args:
              state ([type]): [description]

          Returns:
               (tuple): Tuple containing:

                    action (): The chosen action
                    action_log_prob (float): The log probability of the chosen action
          """
          # if self.dtype == "Images":
          #      state = image_preprocess(state) # Maybe delete?
          
          # Encode the given state of the environment
          enc_state = self.encode_state(state)
          
          # Sample an action from the actor model based on the encoded state
          action_dist = self.actor_model(enc_state)
          dist = Categorical(action_dist)
          action = dist.sample()
          
          #action_log_prob = dist.log_prob(action)
          action_log_prob = torch.squeeze(dist.log_prob(action)).item()
          action = torch.squeeze(action).item()
          
          return action, action_log_prob


     def pred_action(self, state, next_state):
          """ Function for predicting the probability to perform each action by the policy using the inverse  model

          Args:
              state ([type]): The current state
              next_state ([type]): The next state

          Returns:
              [type]: The predicted action based on the specified state and next state
          """
          # Encode the states
          enc_state = self.encode_state(state)
          enc_next_state = self.encode_state(next_state)

          # Predict the action that was performed to result into the new state from the original one
          pred_action = self.inverse_model([enc_state, enc_next_state])

          return pred_action


     def pred_state_value(self,state):
          """[Function for predicting the action performed by the policy using the inverse  model ]
          This function seems to be wrong. Or at least the OG documentation is wrong.
          Args:
               state ([ Gym type]): [state extracted from the environment] . 
               next_state ([ Gym type]): [ next state extracted from the environment] . 
          """
          enc_state = self.encode_state(state)
          value = self.critic_model(enc_state)
          value = torch.squeeze(value).item()
          return value


     # TODO: The following three functions seem to be quite similar. Do we maybe need a class that has specific info for each case?
     # TODO: Should this be the same memory in all cases?
     def store(self, episode, state, next_state, action, action_log_prob, reward, value, done, literals=[], constraint_sat=[]):
          #self.store_rl_info(episode,state,next_state,action,action_log_prob, reward,value, done)
          #self.store_sm_info(state,next_state,action,literals,constraint_sat)
          self.memory.store_memory(episode, state, next_state, action, action_log_prob, reward, value, done, literals, constraint_sat)

     def store_rl_info(self, episode, state, next_state, action, action_log_prob, reward, value, done):
          self.memory.store_memory(episode, state, next_state, action, action_log_prob, reward, value, done)

     def store_sm_info(self,state,next_state,action,literals,constraint_sat):
          self.semantic.memory.store_memory(state, next_state, action, literals, constraint_sat)


##################################################################################################
#Loss functions

     def critic_model_loss(self, enc_state, extrinsic_advantage, extrinsic_value):
          """Loss function for the critic model

          Args:
              enc_state ([type]): Encoding of the current state, used to calculate the critic value.
              extrinsic_advantage (float): The advantage values
              extrinsic_value (float): [description]

          Returns:
              float: The current loss (MSE) for the critic model
          """
          critic_value = self.critic_model(enc_state)
          critic_value = torch.squeeze(critic_value)
          returns = extrinsic_advantage + extrinsic_value

          # Loss calculation - MSE of returns and critic values
          critic_loss = (returns - critic_value) ** 2
          return critic_loss.mean()


     def inverse_model_loss(self, state, next_state, action):
          """Loss function for the inverse model not tested. Cross-entropy of the 

          Args:
              state ([type]): The original state of the environment
              next_state ([type]): The next state of the environment after an action was performed
              action (list[float]): The actual probability of each action 

          Returns:
              float: The cross-entropy loss of the predicted action versus the actual one.
          """
          pred_action = self.pred_action(self, state, next_state)
          return nn.CrossEntropyLoss(pred_action, action)


##################################################################################################
#Advantage functions

     # TODO: This is not implemented.
     def total_advantage(self):
          """[Adds all the advantages if more than one advantage is used not Implemented yet]
          Args:
               arg1 ([type]): [description] . Defaults to
               arg2 ([type]): [description] . Defaults to
               arg3 ([type]): [description] . Defaults to
          """
          return advantage


     def compute_extrinsic_advantage(self, values, rewards, dones):
          """Advantage function calculated using the rewards given by the environment and the value function given by the critic model.

          Args:
              values ([type]): reward batch extracted from the environment
              rewards ([type]): value batch calculated by the critic model
              dones ([type]): completion vector

          Returns:
              float: The calculated advantage.
          """
          # Initial advantage 0.
          advantage = np.zeros(len(rewards), dtype=np.float32)
          
          for t in range(len(rewards) - 1):
               discount = 1
               a_t = 0 

               for k in range(t, len(rewards)-1):
                    a_t += discount * (rewards[k] + self.gamma * values[k+1] \
                         * (1 - int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
               advantage[t] = a_t
          
          return torch.tensor(advantage).to(self.device)


     # TODO: Implement this. This looks like the code from the previous function which may fool us that it is imlemented.
     def intrinsic_world_model_andvantage(self, rewards, values, dones):
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

     # TODO: This also looks to not be implemented
     def advantage_semantic_model(self,rewards,values,dones):
          """[Advantage function for Semantic Utility when is used as a reward not Implemented yet]
          Args:
               arg1 ([type]): [description] . Defaults to
               arg2 ([type]): [description] . Defaults to
               arg3 ([type]): [description] . Defaults to
          """
          return advantage.detach().numpy()


##################################################################################################
#Agent Learning

     # TODO: Errors bellow and the documentation suggest that this is not yet implemented.
     def learn_models_join(self):
          """Learning function for the agent working  but missing some parts when model based or semantic utility 
               is used (finish Implementation only if you already understood the whole code and fill up the missing parts)
          """
          #print("begining:")
          #for name,param in self.semantic_model.named_parameters():
               #if param.requires_grad:
                    #print(name,param.data)
          
          for _ in range(self.learn_epochs):
               episode_arr,state_arr, next_state_arr,action_arr, old_probs_arr,reward_arr,values_arr, dones_arr,literal_arr,const_arr, batches = self.memory.generate_batches()
               advantage= self.compute_extrinsic_advantage(values_arr,reward_arr, dones_arr)
               values = torch.tensor(values_arr).to(self.device)

               # Second Inner loop samples using batch Indexes
               for batch in batches:
                    states = state_arr[batch]
                    enc_states= self.encode_state(states)
                    old_probs = torch.tensor(old_probs_arr[batch]).to(self.device)
                    actions = torch.tensor(action_arr[batch]).to(self.device)
                    
                    # Actor Loss
                    actor_loss = self.policy.ppo_actor_loss(enc_states,actions,old_probs,advantage[batch])
                    
                    # Critic Loss
                    critic_model_loss = self.critic_model_loss(enc_states,advantage[batch],values[batch])
                    critic_model_loss = self.critic_model_loss(enc_states,advantage[batch],values[batch])
                    
                    # Inverse Model Loss
                    if self.use_inverse_model:
                         inverse_model_loss = self.inverse_model_loss([state, next_state], action)
                    else:
                         inverse_model_loss = 0
                    
                    # Semantic Model Loss
                    if self.use_semantic_model:
                         literals = torch.tensor(literal_arr[batch]).to(self.device)
                         literals.long()
                         literals_int = literals.float()
                         # TODO: I do not understand the above. You store a float to a variable with int in the name.

                         # Create constraint and action vectors
                         constraint = torch.tensor(const_arr[batch]).to(self.device)
                         actions = torch.unsqueeze(actions, 1)
                         print("here:", actions)
                         
                         # Calculate the semantic loss
                         semantic_model_loss = self.semantic.deepweights_loss([enc_states, actions], literals_int)
                         # print(semantic_model_loss)
                         # semantic_model_loss = self.semantic.constraint_loss([enc_states,actions],literals_int,constraint)
                    else:
                         semantic_model_loss = 0
                    
                    # World Model Loss
                    if self.use_world_model:
                         forward_model_loss = self.world.forward_model_loss()
                    else:
                         forward_model_loss = 0
                    
                    # Image Model Loss
                    if self.use_image_model:
                         image_model_loss = self.image.image_model_loss()
                    else:
                         image_model_loss = 0
                    
                    # Total Loss
                    loss = actor_loss + 0.5 * critic_model_loss + semantic_model_loss
                    self.agent_optimizer.zero_grad()
                    # address = "gradients/" + self.get_agent_id()
                    # make_dot(loss).render(address, format="png")
                    loss.backward()
                    self.agent_optimizer.step()
                    
          #print("End")
          #for name,param in self.semantic_model.named_parameters():
               #if param.requires_grad:
                    #print(name,param.data)
          self.memory.clear_memory()


     # TODO: The current docuentation suggests that this is not correctly implemented. 
     def learn_models_disjoin(self):
          """[Learning function for the agent working  we generate several datasets and train each models separately [Not Correctly implemented]          """
          for _ in range(self.learn_epochs):
               episode_arr,state_arr, next_state_arr,action_arr, old_probs_arr,reward_arr,values_arr, dones_arr,batches = self.memory.generate_batches()
               state_s_arr,next_state_s_arr,action_s_arr,literal_s_arr,constraint_s_arr,batches_s =self.semantic.memory.generate_batches()
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
                    critic_model_loss = self.critic_model_loss(enc_states,advantage[batch],values[batch])
                    #---------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
                    # Inverse Model Loss
                    if self.use_inverse_model:
                         inverse_model_loss=self.inverse_model_loss([state,next_state],action)
                    else:
                         inverse_model_loss=0
# --------------------------------------------------------------------------------
                    #  Semantic Model Loss
                    if self.use_semantic_model:
                         states_s = state_s_arr[batch]
                         enc_states_s= self.encode_state(states_s)
                         actions_s = torch.tensor(action_s_arr[batch]).to(self.device)
                         literals = torch.tensor(literal_s_arr[batch]).to(self.device)
                         constraint = torch.tensor(constraint_s_arr[batch]).to(self.device)
                         print("here",literals,constraint)
                         semantic_model_loss = self.semantic.deepweights_loss([enc_states_s,actions_s],literals,constraint)
                         #semantic_model_loss = self.semantic.constraint_loss([enc_states,actions],literals_arr,constraint)
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
                    print("loss(actor,critic,deepweights):",actor_loss,critic_model_loss,semantic_model_loss)
                    loss = actor_loss+0.5*critic_model_loss+semantic_model_loss
                    self.agent_optimizer.zero_grad()
                    #address = "gradients/"+self.get_agent_id()
                    #make_dot(loss).render(address, format="png")
                    loss.backward()
                    self.agent_optimizer.step()
          self.memory.clear_memory()
