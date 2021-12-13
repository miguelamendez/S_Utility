"""
    Description: This library contains all the functions/classes for: Agents.
    Please refer to each of the functions/classes for a full description of what they do.

    Functions ([first,second] "first bool" notes the implemented functions "second bool" notes the implemented documentation):
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary


    Classes ([first,second] "first bool" notes the implemented classes "second bool" notes the implemented documentation):
        Agent:[True,False] The class agent is a template for building rl agents , it consist of a dict of graphs that define how models are connected,
        a transition dictionary that stores internal and external information and a memory function used for storing transitions and training the agent.
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        """

import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("build.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# one directory above
#Internal libraries
from ml.agents.config import * #File that contains information of developed models
from ml.models.build import * #File for creating/building a model (only used here build_model func)
from data.utils.data_mngr import *
#External libraries
import numpy as np

class Agent(object):
    """Agent class for building RL agents:
        The class Agent basically is a wrapper for the class models:
        An agent object is composed of a dictionary of models (Nodes), a dictionary of processing functions called structures (Edges) ,
        process graphs build with the Nodes and Edges specifications and a data dictionary that stores a whole transition of data generated by the environment and agent.
    Args:
        object ([type]): [description]
    """

    def __init__(self,models,structures,data_dict,data_mngr=None,name=[],learn_mode=True):
        """[Generic Agent: This wrapper allows us to define an RL agent. """
        self.id=self.set_id(name) #Name of the agent (str)
        self.models=models #Models used by the agent a dictionary of objects (dict of Model objs)
        self.data_dict=data_dict #A dictionary of the data expected to be processed (data_dict={"agent_data":{},"env_data":{})
        self.process_graphs= self.build_graphs(structures) #Processing Graphs usually there is 3 graphs: preinterventional , postinterventional and learning graph
        self.learn_mode=learn_mode #Boolean variable that enables/disables Learning
        if self.learn_mode:
            self.data_mngr=data_mngr
        
##################################################################################################
    def __call__(self,x):
        if flag==0:
            flag=1
            return self.forward(x)
        elif flag==1:
            self.process(x)
        return self.forward(x)

    """[Agent functions are devided in Initialization functions,  processing functions and Memory functions]"""
    def set_id(self,name):
        for model_id in self.models.keys():
            name+="_"+model_id
        return name

    def build_graphs(self,structures):
        graphs={}
        for key in structures.keys():
            graphs[keys]=Graph(self.models,structures[keys])
        return graphs

#Processing functions of agent 
    def start(self,state,train_mode=True):
        #Start is called every new episode
        self.transiton["state"]=state
        self.train=train_mode
        self.memory.reset_buffer()
        #Other initialization stuff required: Probably regarding models
        return

    def act(self,act_id="pre_int"):
        graph=self.process_graphs[act_id]
        data_dict=graph.process(transiton["state"])
        self.store_int_data(data_dict)
        return data_dict["action"]

    def process(self,data_dict,process_id="post_int",learn_id="learn"):
        self.store_ext_data(data_dict)
        graph=self.process_graphs[process_id]
        data_dict=graph.process(self.transition)
        self.store_int_data(data_dict)
        if self.train:
            self.learn(learn_id)
        self.memory.store_transition(self.transition)
        self.transition={}
        self.transition["state"]=state

    def learn(self,learn_id):
        data_dict=self.memory.gen_batch(self.batch_size)
        graph=self.process_graphs[learn_id]
        post_data=graph.process(data_dict)
        post_data=self.post_act_graph.process()
        for model in self.models:
            model.train()

    #Memory Related Functions
    def store_ext_data(self,data_dict):
        for keys in self.data_ids["env"]:
            self.transition[keys]=data_dict[keys]
        #next_state,reward,done=data
        #self.transition["next_state"]=next_state
        #self.transition["reward"]=reward
        #self.transition["done"]=done
        return

    def store_int_data(self,data_dict):
        for keys in self.data_ids["agent"]:
            self.transition[keys]=data_dict[keys]
    #Debbug functions
    def info(self):
        info=[]
        for model in self.models:
            info.append(models.get_specs())



class DummyAgent(object):
    def __init__(self,inputs,outputs,discrete_act=True):
        """Example:
        agent1=DummyAgent(4,[-2,2])
        agent2=DummyAgent(4,[10])
        print(agent1.act())
        print(agent2.act())"""
        self.id ="Dummy"
        self.inputs=inputs
        self.outputs=outputs
        self.output_dic=[]
        self.discrete=self.is_discrete(outputs)
        if discrete_act and not(self.discrete):
            self.discrete_actions()

#Agent Functions:
##################################################################################################
    def start(self,state):
        return

    def is_discrete(self,outputs):
            if len(outputs)>1:
                return False
            else:
                return True
    def discrete_actions(self):
            x=np.arange(self.outputs[0],self.outputs[1],.1)
            x=np.float32(x)
            self.output_dic=dict(list(enumerate(x)))

    def act(self,state):
        if self.discrete:
            x=np.random.randint(self.outputs)
            #print(x[0])
            return x[0]
        else:
            x=len(self.output_dic)
            return [self.output_dic[np.random.randint(x)]]

    def process(self,data):
        return

    def info(self):
        return "Agent that chooses random actions"

def build_graphs(self,structures):
    graphs={}
    for key in structures.keys():
        graphs[keys]=Graph(self.models,structures[keys])
        return graphs


def agent(env_info,env_models,agent_id,int_data_mngr=False):
    """The build_agent function is the main function for building custom agents
    env_info = A dictionary containing the state and action spaces properties [data_type , dimentions, values]
    env_models = A list of ids of the data generated by the environment each timestep
    agent_id= The agents id so it can be build using the agent's config file
    int_data_mngr = Is a boolean variable for the data mngr. If true all training will be done inside the agents object , (An internal data manager will be created)
    """
    #Defining data_ids for total transitions:#############################################################################################
    data_dict={"agent_data":[],"env_data":[]}
    #Setting up env_data.
    for model_id in env_models:
        data_dict["env_data"]={model_id:[]}
    #Setting up agent_data.
    for model_id in agent_dic[agent_id]["models"]:
        data_ids["agent_data"]={model_id:[]}
    #Building Internal Data Manager
    if int_data_mngr:
        data_mngr=agent_dic[agent_id]["data_mngr"]
    #Building Models for nodes:###########################################################################################################
    models={}
    #Building environment models
    for model_id in env_models["models"]:
        model=models[model_id]
        model["model"]=lambda x :torch.tensor(x)
        models["outputs"]=[]
    #Building agent models
    graphs=build_graphs()
    for model_id in agent["models"]:
        model=models[model_id]
        model["model"]=build_model(inputs,outputs,model_id)
        models["outputs"]=[]
    ##############################################################################################################################
    #Building Models for edges:###################################################################################################
    for structure_id in agent["structures"]:
        for edges in structure_id:
            print("hola")
    ##############################################################################################################################
    return Agent(models,structures,data_ids,data_mngr=int_data_mngr,name=[],learn_mode=True)