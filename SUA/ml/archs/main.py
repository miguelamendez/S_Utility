"""
    Description: Main file for archs.
    Currently this file is usless/haven't found any application yet
    """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("main.py",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))


#External libraries
import numpy as np
import torch
import torch.nn as nn

