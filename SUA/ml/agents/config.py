"""
    Description: Config file to build Reinforcement Learning Agents.
    
    """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("agents_config",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))

#External libraries
"""Agents are structures composed with 3 basic elements:
    1st element is a list of deep learning model
    2nd element is a structure that defines the connections between models
"""

envs_dic={
    "gym":{"models":{"state","next_state","reward","done"}
        }
    }
"""
pre-int structure:
post-int structure:
"""
agents_dic={
    "dummy":{
        "models":["rand_action"],
        "structures":{
                    "pre_int":{("state","rand_action"):"dummy_model"},
                    "post_int":None,
                    "learning":None},
        "defaults":{"env_dims":"rand_action","action_dims":"rand_action"}
    },
    "simple_ac":{
        "models":["func_prep","simple_ac"],
        "structures":{"pre_int":{("state","func_prep"):lambda x:x,("func_prep","simple_ac"):lambda x:x},"post_int":{},"learning":{}},
        "defaults":{"env_dims":"rand_action","action_dims":"rand_action"}
    },
    "compose_ac":{"models":["det_rnn_enc","prob_value_nl","prob_policy_ppo"],
        "structures":{"pre_int":{("state","det_rnn_enc"):lambda x:x,("det_rnn_enc","prob_policy_ppo"):lambda x:x.detach()},"post_int":{("next_state","det_rnn_enc"):1,("det_rnn_enc","prob_value_nl"):1},"learning":{("reward","prob_value_nl"):True,("prob_value_nl","prob_policy_ppo"):False}}
    },
    "simple_value":{"models":["prob_rnn_enc","prob_value_nl","prob_policy_ppo_cur" ,"semantic_expert"],
        "structures":{"pre_int":{("state","prob_rnn_enc"),("det_rnn_enc","prob_policy_ppo")},"post_int":{("next_state","det_rnn_enc"),("det_rnn_enc","prob_value_nl"),("det_rnn_enc","semantic_expert")},"learning":{("reward","prob_value_nl"):True,("prob_value_nl","prob_policy_ppo"):False}}
    },
    "simple_dwm":{"models":["det_rnn_enc","prob_value_nl","prob_policy_ppo_cur" ,"prob_prior_trans","prob_post_trans","prob_dec"],
        "structures":{"pre_int":{(0,4):True,(4,6):False},"post_int":{(1,4):True,(4,5):True},"learning":{(2,5):True,(5):True}}
    },
        "simple_su":{"models":["det_rnn_enc","prob_value_nl","prob_policy_ppo_cur","prob_prior_trans","prob_post_trans","prob_dec"],
        "structures":{"pre_int":{(0,4):True,(4,6):False},"post_int":{(1,4):True,(4,5):True},"learning":{(2,5):True,(5):True}}
    }
    }

