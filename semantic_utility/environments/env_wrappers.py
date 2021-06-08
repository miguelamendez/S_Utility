"""[This should be a wrapper for the environments currently not implemented]"""
from semantic_utility.utils.constraint_utils import *
import gym
def semantic_env(env,literals,preprocess_func=[],constraint=[],path=[],name=[]):
    const_flag=False
    if len(constraint)!=0:
        cp=Constraint_Processor(constraint,path,name)
        const_flag=True
    lp1 = Literal_Processor(literals)
    def semantic_wrapper_env(*args, **kwargs):
        extra_data= kwargs.pop('extra_data', None)
        next_state, reward, done, info=env(*args,**kwargs)
        if len(extra_data)!=0:
            if len(preprocess_func)!=0:
                literals_data=preprocess_func(extra_data)
                literals=lp.evaluate_literals(literals_data)
            else:
                literals_data=extra_data
                literals=lp.evaluate_literals(literals_data)
        else:
            if len(preprocess_func)!=0:
                literals_data=preprocess_func(next_state)
                literals=lp.evaluate_literals(literals_data)
            else:
                literals_data=next_state
                literals=lp.evaluate_literals(literals_data)
        if cons_flag:
            constraint_sat=cp.constraint_process(literals)
            return next_state,literals,constraint_sat,reward,done,info
        else:
            return next_state,literals,reward,done,info
return semantic_wrapper_env


self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
