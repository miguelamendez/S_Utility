"""[Main file: used for training Agents : Contains an environment and an agent]
"""

import numpy as np
import gym

from itertools import count
from semantic_utility.utils import *
from semantic_utility.agents.agent_general import *
#from semantic_utility.environments.env_atari import *
#from semantic_utility.environments.env_safety_gym import *

here = Path(__file__).parent

if __name__ == '__main__':
	#Value Environments----------------------------------------
	DATA_TYPE="Values"
	env = gym.make('CartPole-v0')
	
	#Image env------------------------------------------------------
	#DATA_TYPE="RGB_Image_RGB"
	#DATA_TYPE="RGB_Image_GS"
	#env = gym.make('MsPacman-v0')
	#env = AtariPreprocessing(env, frame_skip=1)
	#DATA_TYPE 
	#---------------------------------------------------------------------
	
	state_dim = env.reset()
	dtype, state_dim = format_data_type(state_dim, DATA_TYPE)
	state_dim = state_dim.shape
	#print(dtype)
	#env = AtariPreprocessing(env, frame_skip=1)
	
	"""[Variables]
	"""
	TRAIN_STEPS = 20
	BATCH_SIZE = 5
	N_GAMES = 200
	learn_iters = 0
	avg_score = 0
	n_steps = 0
	#env = gym.make('CartPole-v1')
	agent = Agent(input_dims=4, output_dims=env.action_space.n, dtype=dtype, results=here)
	
	"""[Train ]
	"""
	env_name = env.unwrapped.spec.id
	print(agent.get_agent_id())
	agent_name = agent.get_agent_id()
	figure_file = 'plots/' + env_name + '_' + agent_name + '.png'
	best_score = env.reward_range[0]
	score_history = []


	for i in range(N_GAMES):
		state=env.reset()
		#print(env.observation_space,state.shape)
		done = False
		score = 0

		while not done:
			#Choose Action-----------------------------------------------------------------------
			action, action_log_prob = agent.choose_action(state)
			value = agent.pred_state_value(state)
			next_state, reward, done, info = env.step(action)
			n_steps += 1
			score += reward
			agent.remember(i, state, next_state, action, action_log_prob, reward, value, done)
			
			if n_steps % TRAIN_STEPS == 0:
				agent.learn_model()
				learn_iters += 1
			state = next_state
		
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		if avg_score > best_score:
			best_score = avg_score
		
		print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
		x = [i+1 for i in range(len(score_history))]
	
	plot_learning_curve(x, score_history, figure_file)
