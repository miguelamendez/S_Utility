"""[Main file: used for training Agents : Contains an environment and an agent]
"""

import numpy as np
import gym

from itertools import count
from semantic_utility.utils import *
from semantic_utility.agents.agent_general import *
from semantic_utility.semantic_expert.semantic_expert_envs import *
#from semantic_utility.environments.env_atari import *
#from semantic_utility.environments.env_safety_gym import *

here = Path(__file__).parent


if __name__ == '__main__':
    #Value Environments----------------------------------------
    DATA_TYPE="Values"
    env = gym.make('CartPole-v0')
    #DATA_TYPE 
    #---------------------------------------------------------------------
    state_dim = env.reset()
    dtype, state_dim = format_data_type(state_dim,DATA_TYPE)
    state_dim=state_dim.shape
    """[Variables]
    """
    TRAIN_STEPS = 20
    BATCH_SIZE = 5
    N_GAMES = 3
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    # Expert definition
    literals=[(lambda variables_arr: variables_arr[0]<.5),(lambda variables_arr: variables_arr[0]>-.5),(lambda variables_arr: variables_arr[2]<(5* 2 * 3.1416 / 360)),(lambda variables_arr: variables_arr[2]>-(5 * 2 * 3.1416 / 360))]
    num_literals=len(literals)
    constraint = [[1],[2],[3],[4]]
    s_expert=SemanticExpert(literals=literals,constraint=constraint,path=here,name="cartpole-v0.cnf")

    # Agent definition
    agent = Agent(input_dims=4,output_dims=env.action_space.n,dtype=dtype,path=here,num_literals=num_literals)
    env_name = env.unwrapped.spec.id
    print(agent.get_agent_id())
    agent_name = agent.get_agent_id()
    figure_file = 'plots/'+env_name+'_'+agent_name+'.png'
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
            literals,constraint_sat=s_expert.semantic_process(next_state)
            agent.store(i,state, next_state,action, action_log_prob,reward,value,done,literals,constraint_sat)
            if n_steps % TRAIN_STEPS == 0:
                agent.learn_models_join()
                learn_iters += 1
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,'time_steps', n_steps, 'learning_steps', learn_iters)
        x = [i+1 for i in range(len(score_history))]
    #plot_learning_curve(x, score_history, figure_file)
