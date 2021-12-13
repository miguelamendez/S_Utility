import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("envs_config",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))
#print(os.path.dirname(os.path.dirname(full_path)))

#How to properly define data_type and dimentions of spaces:
#If descrete dtype: 0 (np.int)
#[[4]] One variable of four values
#[[4,4]] 4 variables of four values
#[[2,8,4]] 2*8 variables of four values
#[[3],[4]] 1 variable of 3 values and 1 variable of four values
#[[3],[2,8,4]] 1 variable of 3 values and 2*8 variables of four values
#If continuous dtype: 1 (np.float32)
#[[(-2,2)]] One variable that ranges from (-2,2)
#[[4,(-2,2)]] 4  variables that ranges from (-2,2)
#[[4,(-2,2)],[1,(-3,3)],[2,(-3,2)]] 4  variables that ranges from (-2,2) ,1  variables that ranges from (-3,3) ,2  variables that ranges from (-3,2)
#Line 20 is the start of the dictionary
envs_dic={
        "gym":{"models":["state","next_state","reward","done"],
                "envs":{
                    'CartPole-v0':{"state_space":[[]],"action_space":[[]]},
                    'Acrobot-v1':{"state_space":[[]],"action_space":[[]]},
                    'MountainCar-v0':{"state_space":[[]],"action_space":[[]]},
                    'MountainCarContinuous-v0':{"state_space":[[]],"action_space":[[]]},
                    'Pendulum-v0':{"state_space":[[]],"action_space":[[]]},
                    'BreakoutNoFrameskip-v4':{"state_space":[[3,210, 160, 256]],"action_space":[[16]]},
                    'MsPacmanNoFrameskip-v4':{"state_space":[[3,210, 160, 256]],"action_space":[[16]]},
                    'PongNoFrameskip-v4':{"state_space":[[3,210, 160, 256]],"action_space":[[16]]},
                    'SpaceInvadersNoFrameskip-v4':{"state_space":[[3,210, 160, 256]],"action_space":[[16]]},
                    'MontezumaRevengeNoFrameskip-v4':{"state_space":[[3,210, 160, 256]],"action_space":[[16]]}
                    }
                }
        }




