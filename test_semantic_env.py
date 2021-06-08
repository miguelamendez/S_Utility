"""[]
"""
from itertools import count
import numpy as np
import gym
from semantic_utility.utils import *
from semantic_utility.environments.env_wrappers import *
from semantic_utility.agents.agent_general import *
#from semantic_utility.environments.env_atari import *
#from semantic_utility.environments.env_safety_gym import *
here = Path(__file__).parent
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    l1 = [(lambda variables_arr: variables_arr[0]>-2.4),(lambda variables_arr: variables_arr[1]<2.4),(lambda variables_arr: variables_arr[2]>-(12 * 2 *3.1416/ 360)),(lambda variables_arr: variables_arr[3]<(12 * 2 *3.1416/ 360))]
    @semantic_env(literals=l1,constraint=[[1,2][3,4]],path=here,name="cartpole-v0.cnf")
    env.step()
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation,literals, constraint,reward, done, info = env.step(action)
            print(literals,constraint)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
