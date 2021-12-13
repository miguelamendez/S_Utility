"""
    Description: Config file to build DL Architectures.
    
    """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("archs_config",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))

#Importing architecture libraries.
print("Importing all architecture libraries:")
#SP architectures
from ml.archs.sp import baselines as sp
from ml.archs.sp import asp_baselines as asp

#RL architectures
from ml.archs.rl import su as su

#Classifiers architectures
#from ml.archs.classifiers

#Generator architectures
#from ml.archs.generators

#Predictors architectures
#from ml.archs.predictors

#Other architectures
#from ml.archs.other


#External libraries
import numpy as np
import torch
import torch.nn as nn

data={
    "rl":{
        "ident":{"arch":su.Ident,"parameters":None}
        ,"rand":{"arch":su.Rand,"parameters":None}
        ,"simple_ac":{"arch":su.Simple_FSP_AC,"parameters":{"enc":512,"actor":256,"critic":128},"values":["samples","samples_grad","values"]}
        ,"simple_enc":{"arch":su.Simple_FSP_Enc,"parameters":512}
        ,"squared_image_enc":{"arch":su.Squared_Image_Enc,"parameters":{}}
        ,"rnn":{"arch":su.RNN,"parameters":{}}
        ,"prob_model":{"arch":su.ProbModel,"parameters":{}}
        ,"df_model":{"arch":su.DensFuncModel,"parameters":{}}
        ,"mv_p_model":{"arch":su.MultivarProbModel,"parameters":{}}
        ,"mv_p_model_v2":{"arch":su.MultivarProbModelv2,"parameters":{}}
        ,"simple_edge":{"arch":su.SimpleEdge,"parameters":{}}
    },
    "sp":{
        "sp":{"arch":sp.SP_pytorch,"parameters":{}}
        ,"lsp":{"arch":sp.RSP_pytorch,"parameters":{}}
        ,"rsp":{"arch":sp.RSP_pytorch,"parameters":{}}
        ,"fsp":{"arch":sp.FSP_pytorch,"parameters":{}}
        ,"att_fsp":{"arch":asp.Att_FSP,"parameters":{}}
        ,"2d_fsp":{"arch":asp.FSP_2D,"parameters":{}}
        ,"mv_fsp":{"arch":asp.MultivarFSP,"parameters":{}}
        ,"att_lsp":{"arch":None,"parameters":{}}
        ,"2d_lsp":{"arch":None,"parameters":{}}
        ,"mv_lsp":{"arch":None,"parameters":{}}
    },
    "classifiers":{
    },
    "generators":{
    },
    "predictors":{
        "GRU":{"arch":nn.GRUCell,"parameters":[]}
    },
    "dummy":{
        "ident":{"arch":lambda x: torch.tensor(x),"parameters":{}},
        "rand":{"arch":lambda x:torch.tensor(x),"parameters":{}}
    }
}
