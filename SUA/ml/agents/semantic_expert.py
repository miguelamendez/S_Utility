"""[This is the file that defines the semantic expert]"""
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("agent_se.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))# one directory above
from utils.symbolic.logic import *

class SemanticExpert_old():
    def __init__(self,literals,preprocess_func=[],constraint=[],path=[],name=[]):
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
        if self.prep_func_flag:
            literals_data=self.preprocess_func(extra_data)
            literals=self.lp.evaluate_literals(literals_data)
        else:
            literals_data=extra_data
            literals=self.lp.evaluate_literals(literals_data)
        if self.const_flag:
            constraint_sat=self.cp.wmc(literals)
            return literals,constraint_sat
        else:
            return literals
  
class SemanticExpert():
    def __init__(self,literals,constraint,constraint_path=[],preprocess_func=[]):
        self.prep_func_flag=False
        self.cp=Constraint_Processor(constraint,path)
        if len(preprocess_func)!=0:
            self.preprocess_func=preprocess_func
            self.prep_func_flag=True
        self.lp = Literal_Processor(literals)
        self.l_val=[] #Stores literals value
    def literal_process(self,data):
        if self.prep_func_flag:
            literals_data=self.preprocess_func(data)
            self.l_val=self.lp.evaluate_literals(literals_data)
        else:
            self.l_val=self.lp.evaluate_literals(data)
        return self.l_val

    def constraint_process(self,data=[]):
            if data==[]:
                return self.cp.wmc(self.l_val)
            else:
                return self.cp.wmc(data)
