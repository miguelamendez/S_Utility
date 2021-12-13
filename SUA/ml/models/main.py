"""Main file for trianing Models"""
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("Printing all Internal libraries:")
print("######################################################################")
print("main.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# one directory above
#Internal libraries:
#Import main class for models and config file
from ml.models.build import *
from ml.models import config as models_cf

#Import data dic from datasets
from data.datasets import config as data_cf
from data.datasets.build import dataset as dataset_build
#from data.datasets.image import config as image_cf
#from data.datasets.text import config as text_cf
#from data.datasets.audio import config as audio_cf
#from data.datasets.tabular import config as tabular_cf
#from data.datasets.other import config as other_cf

#Libraries for training
#Import training functions
from ml.functional.optimizers import *
from ml.functional.obj_fn import *
from ml.functional.train_fn import* 

#Import data preprocessing functions
from data.utils.preprocessing import *

print("End of Internal Libraries###########################################\n")

#External libraries
import warnings
warnings.simplefilter("ignore", UserWarning)
import cv2 as cv
import matplotlib.pyplot as plt

#Some information of the environments

#Print information of availble datasets:
print("Available datasets :\n",list(data_cf.data.keys()))
#Print information of available agents:
print("Avalilable models : \n",list(models_cf.data.keys()))
print("\n")
#Atari preprocessing wrapper

def work(dataset_id,model_id,data_preproc=False,train=False):
    #Getting data set
    data_set=get_dataset(dataset_id)
    #Generating data_mngr with sataset
    data_mngr, data_mngr_info, data_dims = build_data_mngr(data_set)
    print("Env Info:",dara_mngr_info)
    print("Data dims:",data_dims)
    #Setting Model
    print("####################################################")
    print("Building Model:")
    model=model_build(env_info,env_models,agent_id,int_data_mngr=True)
    print("Model name:",model.id)
    print("Model info:",model.specs())
    
    #Running train/test function
    if train:
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
#work(dataset_id='',model_id="dummy",data_preproc=True,train=False)
