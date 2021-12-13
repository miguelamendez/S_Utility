"""
    Description: Config file to build DL Models.
    
    """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("config",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))

#External libraries
import torch.nn as nn
import torch.optim as optim
"""Models are structures composed with 3 basic elements:
    1st element is the architecture which is composed of
    Template:
            "model_id":{
            "arch":{"type":"arch_type","id":"arch_name","inout":[]},
            "obj_fn":{"fn":obj/fn,"params":None},
            "optim":{"fn":obj/fn,"params":None},
            "train_fn":{"fn":obj/fn,"params":None}
                }
"""
data={
    "agent_nodes":{
        "simple_rand":{
            "arch":{"type":"rl","id":"fsp","inout":[]},
            "obj_fn":{"id":"obj_name","params":[]},
            "optim":{"id":"optim_name","params":[]},
            "train_fn":{"id":"train_name","params":[]}
                },
        "simple_ac":{
            "arch":{"type":"rl","id":"simple_ac","inout":{"actor":[32,18],"critic":[32,1],"enc":[128,32]}},
            "obj_fn":{"id":nn.NLLLoss,"params":[]},
            "optim":{"id":optim.Adam,"params":{"lr":1e-3}},
            "train_fn":{"id":lambda x:x,"params":[]}
                },
        "det_enc":{
            "arch":{"type":"arch_type","id":"arch_name","inout":[]},
            "obj_fn":{"id":"obj_name","params":[]},
            "optim":{"id":"optim_name","params":[]},
            "train_fn":{"id":"train_name","params":[]}
                },
        "prob_pol":{
            "arch":{"type":"arch_type","id":"arch_name","inout":[]},
            "obj_fn":{"id":"obj_name","params":[]},
            "optim":{"id":"optim_name","params":[]},
            "train_fn":{"id":"train_name","params":[]}
                },
        "prob_value":{
            "arch":{"type":"arch_type","id":"arch_name","inout":[]},
            "obj_fn":{"id":"obj_name","params":[]},
            "optim":{"id":"optim_name","params":[]},
            "train_fn":{"id":"train_name","params":[]}
                },
        "prob_state":{
            "arch":{"type":"arch_type","id":"arch_name","inout":[]},
            "obj_fn":{"id":"obj_name","params":[]},
            "optim":{"id":"optim_name","params":[]},
            "train_fn":{"id":"train_name","params":[]}
                }
        },
    "agent_edges":{
        "simple_edge":{
            "arch":{"type":0,"id":"","inout":[]},
            "obj_fn":None,
            "optim":None,
            "train_fn":None
        },
    "dummy_model":{
            "arch":{"type":0,"id":"","inout":[]},
            "obj_fn":None,
            "optim":None,
            "train_fn":None
        }
    },
    "classifiers":{
        "model_id":{
            "arch":{"type":"arch_type","id":"arch_name","inout":[]},
            "obj_fn":{"id":"obj_name","params":[]},
            "optim":{"id":"optim_name","params":[]},
            "train_fn":{"id":"train_name","params":[]}
                }
        },
    "predictors":{
        "credit_sp":{
            "arch":{"type":"arch_type","id":"arch_name","inout":[]},
            "obj_fn":{"id":"obj_name","params":[]},
            "optim":{"id":"optim_name","params":[]},
            "train_fn":{"id":"train_name","params":[]}
                }
        },
    "generators":{
        "model_id":{
            "arch":{"type":"arch_type","id":"arch_name","inout":[]},
            "obj_fn":{"id":"obj_name","params":[]},
            "optim":{"id":"optim_name","params":[]},
            "train_fn":{"id":"train_name","params":[]}
                }
    }
    }

