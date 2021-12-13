"""
    Description: This library contains all the functions/classes for: training with dynamic data (e.g reinforcement learning environments)
    Please refer to each of the functions/classes for a full description of what they do.

    Functions ([first,second] "first bool" notes the implemented functions "second bool" notes the implemented documentation):
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary


    Classes ([first,second] "first bool" notes the implemented classes "second bool" notes the implemented documentation):
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("dynamic.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path))) # one directory above


#External libraries
import torch
import numpy as np

def train(agent,env,ext_data_mngr=None,pre_proc_fn=None,episodes=16):
    #This is the high-level function for training agents:
    #Cheking valid attributes:
    if len(pre_proc_fn)>1:
        raise ValueError("Preprocessing functions for multiple state_spaces not supported")
    else:
        pre_proc=pre_proc_fn["space_0"]
    if ext_data_mngr==None:
        print("Training will be done using internal agent data_mngr")
        if agent.data_mngr==None:
            raise ValueError("Agent cannot be trained:Not internal or external data manager")
    #Initializing external training dictionary
    train_dict={}
    #Main train loop:
    for episode in range(0,episodes):
        state=env.reset()
        state=pre_proc(state) if pre_proc is not None else state #Preprocessing initial state
        agent.start(torch.tensor(state)) #Initializing agent (some agents require some initial values to start doing training and inference also it resets the internal memory)
        done = False
        score = 0
        n_steps = 0
        episode_data=[]
        while not done:
            #Choose Action-----------------------------------------------------------------------
            action = agent.act(torch.tensor(state)) #Agent acts
            next_state, reward, done, info = env.step(action) #Environment step 
            next_state = pre_proc(next_state) if pre_proc is not None else next_state #Preprocessing next state if there is a preprocessing function
            data_dict={"next_state":torch.tensor(next_state), "rewards":torch.tensor(reward), "done":torch.tensor(done)} #Creating data_dict
            ext_data_mngr.store(data_dict) if ext_data_mngr is not None else agent.store(data_dict)#Storing data_dict
            learned_data=agent.learn(ext_data_mngr())
            agent.process(data_dict) #Agent processing function
            n_steps += 1
            score += reward
            episode_data.append([n_steps,score])
        train_dict[episode]=episode_data
    env.close()
    return train_dict

def test(agent,env,pre_proc_fn=None,episodes=4,render=False):
    #Cheking valid attributes:
    if len(pre_proc_fn)>1:
        raise ValueError("preprocessing functions for multiple state_spaces not supported")
    else:
        pre_proc=pre_proc_fn["space_0"]
    #Initializing external training dictionary
    test_dict={}
    #Main train loop:
        #Main train loop:
    for episode in range(0,episodes):
        state=env.reset()
        state=pre_proc(state) if pre_proc is not None else state #Preprocessing initial state
        agent.start(torch.tensor(state)) #Initializing agent (some agents require some initial values to start doing training and inference also it resets the internal memory)
        done = False
        score = 0
        n_steps = 0
        episode_data=[]
        while not done:
            if render:
                env.render()
            #Choose Action-----------------------------------------------------------------------
            action = agent.act(torch.tensor(state))
            next_state, reward, done, info = env.step(action)
            next_state = pre_proc(next_state) if pre_proc is not None else next_state
            data_dict={"next_state":torch.tensor(next_state), "reward":torch.tensor(reward), "done":torch.tensor(done)}
            agent.process(data_dict) #agent.process()
            n_steps += 1
            score += reward
            episode_data.append([n_steps,score])
        test_dict[episode]=episode_data
    env.close()
    return test_dict
