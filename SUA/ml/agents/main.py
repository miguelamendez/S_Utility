"""Main file for trianing RL agents"""
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("Printing all Internal libraries:")
print("######################################################################")
print("main.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# one directory above
#Internal libraries:
#Import main class for agents and config file
from ml.agents.build import *
from ml.agents import config as cf

#Import main class for environments and config file
from data.envs.build import *
from data.envs import config as cf

#Import training functions
from ml.train.dynamic import *

#Import data preprocessing functions (for external preprocessing)
from data.utils.preprocessing import *

print("End of Internal Libraries###########################################\n")
#External libraries
import warnings
warnings.simplefilter("ignore", UserWarning)
import cv2 as cv
import matplotlib.pyplot as plt

#Some information of the environments


#Print information of availble envs:
print("Available envs from Gym :\n",list(envs_dic["gym"]["envs"].keys()))
#Print information of available agents:
print("Avalilable agents : \n",list(agents_dic.keys()))
print("\n")
#Atari preprocessing wrapper

def work(env_id,agent_id,ext_preproc=False,train=False,episodes=4,render=False):
    
    #Setting environment and getting appropriate information:
    print("Building Environment:")
    env , env_info, env_models = build_env(env_id) #env is the environment object Example: if using gym will be same as env=gym.make("env_id")
    #env_info is a dictionary containing state_dims ,action_dims, state_dtype , actions_dtype
    print("Environment name:",env_id) #Getting the id of the environments
    print("Env Info:",env_info) #Getting dimentions of env and action spaces
    print("Env Models:",env_models) #Getting the data that can be processed from the environment gym: next_state ,reward , done 
    
    #Defining external preprocessing functions for state_spaces.
    if ext_preproc:
        pre_proc={}
        for space in env_info["state_space"]:
            #If state space are colored Images (i.e, env_info["state_space"]["space_0"]["vals"]=256 and env_info["state_space"]["dims"]==[3,210, 160])
            if env_info["state_space"][space]["vals"]==256 and env_info["state_space"][space]["dims"]==[3,210, 160]:
                pre_proc[space]=(Image(resize=(128,128),grayscale=False))
            else:
                pre_proc[space]=None
        print("External preprocessing:\n",pre_proc)

    #Setting agent
    print("####################################################")
    print("Building Agent:")
    #agent=build_agent(env_info,env_models,agent_id,int_data_mngr=True) #Calling build_agent function
    agent=DummyAgent(inputs=[128,128],outputs=[16])
    print("Agent id:",agent.id)
    print("Agent info:",agent.info())
    #Running train/test function
    if not test_mode:
        print("\n####################################################")
        print("Training Mode:")
        data=train(agent,env,ext_data_mngr,pre_proc_fn=pre_proc,episodes=episodes)
    else:
        print("\n ####################################################")
        print("Testing Mode:")
        data=test(agent,env,pre_proc_fn=pre_proc,episodes=episodes,render=render)
        #Printing final score
        for i in data.keys():
            print(data[i][-1:])
print("Initializing work function ########################################################")
work(env_id='MsPacmanNoFrameskip-v4',agent_id="dummy",ext_preproc=True,episodes=2,render=False)
