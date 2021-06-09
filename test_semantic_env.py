"""[]
"""
from itertools import count
import numpy as np
import gym
from semantic_utility.utils import *
from semantic_utility.semantic_expert.semantic_expert_envs import *
from semantic_utility.agents.agent_general import *
#from semantic_utility.environments.env_atari import *
#from semantic_utility.environments.env_safety_gym import *
here = Path(__file__).parent
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    l1=[(lambda variables_arr: variables_arr[0]<2.4),(lambda variables_arr: variables_arr[0]>-2.4),(lambda variables_arr: variables_arr[2]<(12* 2 * 3.1416 / 360)),(lambda variables_arr: variables_arr[0]>-(12 * 2 * 3.1416 / 360))]
    #l1 = [(lambda variables_arr: variables_arr[0]<1),(lambda variables_arr: variables_arr[0]>-1),(lambda variables_arr: variables_arr[2]<(10 * 2 * 3.1416 / 360)),(lambda variables_arr: variables_arr[2]>-(10 * 2 * 3.1416 / 360)),(lambda variables_arr: abs((variables_arr[1]-variables_arr[3]))<5)]
    constraint = [[1],[2],[3],[4]]
    se=SemanticExpert(literals=l1,constraint=constraint,path=here,name="cartpole-v0.cnf")
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            literals,constraint_sat=se.semantic_process(observation)
            print(literals,constraint_sat)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
