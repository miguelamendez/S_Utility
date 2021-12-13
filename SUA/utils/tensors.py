"""
    Description: This library contains all the functions/classes for: operations with tensors (pytorch)
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
print("tensor.py:",full_path)
sys.path.append(os.path.dirname(full_path))# one directory above
#print(os.path.dirname(full_path))
#from dir.to.file import * #files for imports
#from dir.to.file import *

#External libraries
import numpy as np
import matplotlib.pyplot as plt
import torch 
import gym

def add_dimention(tensor):
	"""[Function that allows a single tensor to be processed by the neural networks that use Sequential function]

	Args:
		tensor ([type]): [single tensor whitout the batch dimention]

	Returns:
		tensor[type]: [returns a tensor with extra dimention so it can be prosseced by Sequential in pytorch]
	"""    
	tensor = tensor.unsqueeze(0) #Add extra dimention tensor.double() is equivalent to tensor.to(torch.float64)
	tensor = tensor.double()  #Formats tensor into double type
	return tensor

def list2tensors(some_list):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: list
        - Output: list of tensors

    Examples::

    """
    t_list=[]
    for i in some_list:
        t_list.append(torch.tensor(i))
    return t_list



