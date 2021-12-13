"""
    Description: This library contains all the class model for building ml architectures
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
print("build.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# one directory above

from ml.archs.config import data as archs_dict #File that contains dictionary of implemented architectures


#External libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#A simple class for transforming functions into nn.Modules
class Arch(nn.Module):
    def __init__(self,func):
        super(Arch, self).__init__()
        self.func=func

    def forward(self, x):
        return self.func(x)
    

    """Generally speaking, this means, that we can define a class A with type(classname, superclasses, attributedict)
            When we call "type", the call method of type is called. The call method runs two other methods: new and init:
            type.__new__(typeclass, classname, superclasses, attributedict)
            type.__init__(cls, classname, superclasses, attributedict)"""
#Example of Arch
#arch=Arch(lambda x:2*x)
#x=torch.randn([2,2])
#print(x,"\n",arch(x))

def arch(arch_type,arch_id,inout,parameters=None,func=False):
    archs=archs_dict[arch_type]
    arch_data= archs[arch_id]
    #Compiling a functional architecture
    if func:
        arch=Arch(func)
    else:
        if bool(parameters) :
            arch = arch_data["arch"](inout,parameters)
        else:
            arch = arch_data["arch"](inout,arch_data["parameters"])
    return arch

#Example of arch:
#new_arch=arch("rl","ident",[3,2])
#x=torch.randn([2,2])
#print(new_arch(x).size())

