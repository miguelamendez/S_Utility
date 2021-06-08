"""[This should be a wrapper for the environments currently not implemented]"""
from semantic_utility.utils.constraint_utils import *
import gym
class SemanticExpert():
    def __init__(env,literals,preprocess_func=[],constraint=[],path=[],name=[]):
        self.const_flag=False
        self.prep_func_flag=False
        if len(constraint)!=0:
            self.cp=Constraint_Processor(constraint,path,name)
            self.const_flag=True
        if len(preprocess_func)!=0:
            self.preprocess_func=preprocess_func
            self.prep_func_flag=True
        self.lp = Literal_Processor(literals)
    def semantic_process(self,extra_data):
        if prep_func_flag:
            literals_data=self.preprocess_func(extra_data)
            literals=self.lp.evaluate_literals(literals_data)
        else:
            literals_data=extra_data
            literals=self.lp.evaluate_literals(literals_data)
        if cons_flag:
            constraint_sat=self.cp.constraint_process(literals)
            return literals,constraint_sat
        else:
            return literals
  
