import random
import numpy as np
from collections import namedtuple

#Replay Memory for Agent
class ReplayMemoryAgent:
    def __init__(self, batch_size):
        """[Replay Memory used by the agent: Implemented ]
        Args:
            batch_size ([type]): [size of the batches used when called by the agent learning function]
        """     
        self.episodes = []   
        self.states = []
        self.next_states = []
        self.actions = []
        self.actions_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        """[Function that samples batches on size batch from the stored memory]

        Returns:
            [type]: [batch of transitions from the dynamics of the environment]
        """        
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.episodes),\
                np.array(self.states),\
                np.array(self.next_states),\
                np.array(self.actions),\
                np.array(self.actions_log_probs),\
                np.array(self.rewards),\
                np.array(self.values),\
                np.array(self.dones),\
                batches

    def store_memory(self, episode,state, next_state,action, action_log_prob,reward, value, done):
        """[summary]

        Args:
            state ([type]): [description]
            action ([type]): [description]
            probs ([type]): [description]
            vals ([type]): [description]
            reward ([type]): [description]
            done (function): [description]
        """        
        self.episodes.append(episode)
        self.states.append(state)
        self.actions.append(action)
        self.actions_log_probs.append(action_log_prob)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear_memory(self):
        self.episodes = []
        self.states = []
        self.next_states = []
        self.actions = []
        self.actions_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

#Replay Memory for Agent on Safety G	ym environments
class ReplayMemoryAgentSafety:
    def __init__(self, batch_size):
        """[Replay Memory used by the agent: Implemented ]
        Args:
            batch_size ([type]): [size of the batches used when called by the agent learning function]
        """     
        self.episodes = []   
        self.states = []
        self.next_states = []
        self.actions = []
        self.actions_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.literals =[]

        self.batch_size = batch_size

    def generate_batches(self):
        """[Function that samples batches on size batch from the stored memory]

        Returns:
            [type]: [batch of transitions from the dynamics of the environment]
        """        
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.episodes),\
                np.array(self.states),\
                np.array(self.next_states),\
                np.array(self.actions),\
                np.array(self.actions_log_probs),\
                np.array(self.rewards),\
                np.array(self.values),\
                np.array(self.dones),\
                np.array(self.literals),\
                batches

    def store_memory(self, episode,state, next_state,action,action_log_prob, reward, value, done, literals):
        """[summary]

        Args:
            state ([type]): [description]
            action ([type]): [description]
            probs ([type]): [description]
            vals ([type]): [description]
            reward ([type]): [description]
            done (function): [description]
        """        
        self.episodes.append(episode)
        self.states.append(state)
        self.actions.append(action)
        self.actions_log_probs.append(action_log_prob)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.literals.append(literals)

    def clear_memory(self):
        self.episodes = []
        self.states = []
        self.next_states = []
        self.actions = []
        self.actions_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.literals = []

#Replay Memory for Policy Model
class ReplayMemoryPolicyModel:
    def __init__(self, batch_size):
        """[Implemented  but maybe missing parts]

        Args:
            batch_size ([type]): [description]
        """        
        self.enc_states = []
        self.enc_next_states = []
        self.action_probs = []
        self.extrinsic_vals = []
        self.extrinsic_advantages = []

        self.batch_size = batch_size

    def generate_batches(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.enc_states),\
                np.array(self.enc_next_states),\
                np.array(self.action_probs),\
                np.array(self.critic_vals),\
                np.array(self.advantages),\
                batches

    def store_memory(self, enc_state, enc_next_state, action_dist, extrinsic_val, extrinsic_advantage):
        """[summary]

        Args:
            state ([type]): [description]
            action ([type]): [description]
            probs ([type]): [description]
            vals ([type]): [description]
            reward ([type]): [description]
            done (function): [description]
        """        
        self.enc_states.append(enc_state)
        self.enc_next_states.append(enc_next_state)
        self.action_probs.append(action_dist)
        self.extrinsic_vals.append(extrinsic_val)
        self.extrinsic_advantages.append(extrinsic_advantage)

    def clear_memory(self):
        self.enc_states = []
        self.enc_next_states = []
        self.action_probs = []
        self.extrinsic_vals = []
        self.extrinsic_advantages = []

#Replay Memory for World Model
class ReplayMemoryWorldModel:
    def __init__(self, batch_size,model_type):
        """[Not Implemented correctly]

        Args:
            batch_size ([type]): [description]
            model_type ([type]): [description]
        """        
        self.pred_phi1_states = []
        self.batch_size = batch_size

    def generate_batches(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        """[summary]

        Args:
            state ([type]): [description]
            action ([type]): [description]
            probs ([type]): [description]
            vals ([type]): [description]
            reward ([type]): [description]
            done (function): [description]
        """        
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        
#Replay Memory for Semantic Model:
class ReplayMemorySemanticModel:
    def __init__(self, batch_size):
        """[Not Implemented correctly]

        Args:
            batch_size ([type]): [description]
        """        
        self.phi1_state = []
        self.phi2_state = []
        self.action = []
        self.action_log_probs = []
        self.action_hat = []
        self.action_hat_log_probs = []
        self.batch_size = batch_size

    def generate_batches(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        n_states = len(self.phi1_state)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.phi1_state),\
                np.array(self.phi2_state),\
                np.array(self.action),\
                np.array(self.action_log_probs),\
                np.array(self.action_hat),\
                 np.array(self.action_hat_log_probs),\
                batches

    def store_memory(self,phi1,phi2,action,action_log_probs,action_hat,action_hat_log_probs):
        """[summary]

        Args:
            phi1 ([type]): [description]
            phi2 ([type]): [description]
            action ([type]): [description]
            action_log_probs ([type]): [description]
            action_hat ([type]): [description]
            action_hat_log_probs ([type]): [description]
        """        
        self.phi1_state.append(phi1)
        self.phi2_state.append(phi2)
        self.action.append(action)
        self.action_log_probs.append(action_log_probs)
        self.action_hat.append(action_hat)
        self.action_hat_log_probs.append(action_hat_log_probs)

    def clear_memory(self):
        self.phi1_state = []
        self.phi2_state = []
        self.action = []
        self.action_log_probs = []
        self.action_hat = []
        self.action_hat_log_probs = []
        
#Replay Memory for Feature Encoder          
class ReplayMemoryImageModel:
    def __init__(self, batch_size):
        """[Not Implemented correclty OPTIONAL ommit for semantic utility paper]

        Args:
            batch_size ([type]): [description]
        """        
        self.pred_phi1_states = []

        self.batch_size = batch_size

    def generate_batches(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        """[summary]

        Args:
            state ([type]): [description]
            action ([type]): [description]
            probs ([type]): [description]
            vals ([type]): [description]
            reward ([type]): [description]
            done (function): [description]
        """        
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []      

#Dont Recomend to touch
class ReplayMemoryGeneral:
	def __init__(self, batch_size):
		"""[Optional: Not correctly implemented]

		Args:
			batch_size ([type]): [description]
		"""	
		#Environment
		self.states = []
		self.next_states = []
		self.rewards = []
		self.dones = []
		#Policy Model
		self.actions_log_probs = []
		self.vals = []
		self.actions = []
		#Feature_Encoder
		self.pred_phi1_states = []
		self.mu_phi1_states = []
		self.std_phi1_states = []
		self.pred_phi2_states = []
		self.mu_phi2_states = []
		self.std_phi2_states = []
		#Forward Model
		self.pred_phi2_hat_states = []
		self.mu_phi2_hat_states = []
		self.std_phi2_hat_states = []
		#Inverse Model
		self.actions_hat_log_probs = []
		self.actions_hat = []

	def generate_batches(self):
		"""[summary]

		Returns:
			[type]: [description]
		"""	
		n_states = len(self.states)
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype=np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]

		return np.array(self.states),\
				np.array(self.next_states),\
				np.array(self.rewards),\
				np.array(self.dones),\
				np.array(self.actions),\
				np.array(self.actions_log_probs),\
				np.array(self.vals),\
				np.array(self.pred_phi1_states),\
				np.array(self.mu_phi1_states),\
				np.array(self.std_phi1_states),\
				np.array(self.pred_phi2_states),\
				np.array(self.mu_phi2_states),\
				np.array(self.std_phi2_states),\
				np.array(self.pred_phi2_hat_states),\
				np.array(self.mu_phi2_hat_states),\
				np.array(self.std_phi2_hat_states),\
				np.array(self.actions_hat_log_probs),\
				np.array(self.actions_hat),\
				batches

	def store_memory(self, state,next_state, action, action_log_probs, vals, reward, done,pred_phi1_state=-1,mu_phi1_state=-1,std_phi1_state=-1,pred_phi2_state=-1,mu_phi2_state=-1,std_phi2_state=-1,pred_phi2_hat_state=-1,mu_phi2_hat_state=-1,std_phi2_hat_state=-1,action_hat=-1, action_hat_log_probs=-1):
        #Environment
		self.states.append()
		self.next_states.append()
		self.rewards.append()
		self.dones.append()
		#Policy Model
		self.actions_log_probs.append()
		self.vals.append()
		self.actions.append()
		#Feature_Encoder
		self.pred_phi1_states.append()
		self.mu_phi1_states.append()
		self.std_phi1_states.append()
		self.pred_phi2_states.append()
		self.mu_phi2_states.append()
		self.std_phi2_states.append()
		#Forward Model
		self.pred_phi2_hat_states.append()
		self.mu_phi2_hat_states.append()
		self.std_phi2_hat_states.append()
		#Inverse Model
		self.actions_hat_log_probs.append()
		self.actions_hat.append()

	def clear_memory(self):
		#Environment
		self.states = []
		self.next_states = []
		self.rewards = []
		self.dones = []
		#Policy Model
		self.actions_log_probs = []
		self.vals = []
		self.actions = []
		#Feature_Encoder
		self.pred_phi1_states = []
		self.mu_phi1_states = []
		self.std_phi1_states = []
		self.pred_phi2_states = []
		self.mu_phi2_states = []
		self.std_phi2_states = []
		#Forward Model
		self.pred_phi2_hat_states = []
		self.mu_phi2_hat_states = []
		self.std_phi2_hat_states = []
		#Inverse Model
		self.actions_hat_log_probs = []
		self.actions_hat = []
