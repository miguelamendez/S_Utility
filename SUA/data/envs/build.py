import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("env.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# one directory above
#Internal libraries
from data.envs.config import * #File that contains information of developed models
#External libraries
import numpy as np
import gym


def get_wrapper(env_id):
    for wrapper in envs_dic:
        for env in envs_dic[wrapper]["envs"]:
            #print(env)
            if env == env_id:
                return wrapper
    return False


def get_space_info(d_list):
    space_info={}
    for idx,var in enumerate(d_list):
        space_name = "space_"+str(idx)
        space_info[space_name]={}
        if isinstance(var[-1], tuple):
            space_info[space_name]["d_type"]=1
            space_info[space_name]["vals"]=var[-1]
        if isinstance(var[-1], int):
            space_info[space_name]["d_type"]=0
            space_info[space_name]["vals"]=var[-1]
        if var[:-1]==[]:
            space_info[space_name]["dims"]=[1]
        else:
            space_info[space_name]["dims"]=var[:-1]
    return space_info
#Example
#space_info=get_space_info([[2,1],[2,1,4]])
#print(space_info)

def build_env(env_id):
    wrapper=get_wrapper(env_id)
    env_info={"state_space":[],"action_space":[]}
    if not wrapper:
        raise ValueError("Environment doesnt exist in wrapper dictionary")
    elif wrapper=="gym":
        env=gym.make(env_id)
        env_models=envs_dic[wrapper]["models"]
        env_info["state_space"] = get_space_info(envs_dic[wrapper]["envs"][env_id]["state_space"])
        env_info["action_space"] = get_space_info(envs_dic[wrapper]["envs"][env_id]["state_space"])
        return env , env_info , env_models
    elif wrapper!="gym":
        #env=other_env.make()
        #env_models=envs_dic[wrapper]["models"]
        #env_info["state_space"] = get_space_info(envs_dic[wrapper]["envs"][env_id]["state_space"])
        #env_info["action_space"] = get_space_info(envs_dic[wrapper]["envs"][env_id]["state_space"])
        raise ValueError("Build function not implemented for: ",wrapper)
    
#Example
#env,env_info,env_models=build_env("BreakoutNoFrameskip-v4")
#print(env_info["state_space"])

def get_gym_spaces(env):
        #This function should extract the same information that is given in the envs config file.
        #Its not implemented yet
        state_space=env.observation_space
        print(state_space)
        action_space=env.action_space
        print(action_space)

#Example
#env=gym.make("BreakoutNoFrameskip-v4")
#get_gym_spaces(env)
