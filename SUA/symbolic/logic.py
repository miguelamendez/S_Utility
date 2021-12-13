"""
    Description: This library contains all the functions/classes for: processing propositional logic constraints and literals
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
print("logic.py:",full_path)
sys.path.append(os.path.dirname(full_path)) # one directory above
from utils.cnf import *
from symbolic.sat import *

#External libraries
from pathlib import Path
import math
from pysdd.sdd import SddManager, Vtree, WmcManager
import os

class Literal_Processor(object):
    def __init__(self,literal_arr):
        self._literals_arr = literal_arr
    
    def evaluate_literals(self,variables_arr):
        """[This function checks whether the propositions defined over some ranges are satisfied (called literals).Then it generates a returns a list, the entries of which correspond to satisfaction of each proposition.]
        Returns:
            List: A list of booleans corresponding to satisfaction of each literal
        """
        literal_satisfactions = []
        for i in self._literals_arr:
            literal_satisfactions.append(i(variables_arr))
        return literal_satisfactions


class Constraint_Processor(object):
    def __init__(self, formula=[],path):
        """[summary]
            formula = Propositional logic constraint a.k.a Theory 
            formula_name =name of the propositional formula Example theory1.cnf
            variables_arr = a list of variables observed by the agent 
            literal_arr = is an array of functions usually lambdas defined over the variables_arr
            Example:l1 = [(lambda variables_arr: variables_arr[0]<0),(lambda variables_arr: variables_arr[0]>1),(lambda variables_arr: 0<=variables_arr[0]<=1),(lambda variables_arr: variables_arr[0]==1),(lambda variables_arr: variables_arr[0]==variables_arr[1])]
        """
        #print(path)
        self.cnf_format =CNF_Formats()
        self.cnf_prcss = CNF_processing()
        #self.name_cnf=
        #self.name_ssd=
        if len(formula)!=[]:
            name_cnf=
            formula_path=path+formula_name+".cnf"
            print(formula_path)
            #self.formula = self.formula_to_cnf_list(formula) #Needs to be implemented 
            self.formula = formula
            num_vars , num_clauses = self.cnf_format.list_to_dimacs_cnf(formula,path) 
            self.sdd_mgr, self.sdd_node= SddManager.from_cnf_file(bytes(path / formula_name))
        else:
            self.sdd_mgr, self.sdd_node= SddManager.from_cnf_file(bytes(path / "base_theory.cnf"))
        name= formula_name.split(".")
        sdd_dir=str(path)+"/SDD/sdd_"+name[0]+".dot"
        with open(sdd_dir, "w") as out:
            print(self.sdd_mgr.dot(), file=out)
    def sat(self,constraint):
    	return
    def wmc(self, weights):
        """[summary]
        Returns WMC of a propositional constraint of an SDD manager.
        Args:
            weights ([type]): The weight of each literal's satisfaction. Expecting an Numpy array but will also work with a python list.
        """
        
        #set weights
        #lits = [None] + [self._sdd_mgr.literal(i) for i in range(1, _sdd_mgr.var_count() + 1)]
        wmc = self.sdd_node.wmc(log_mode=False)
        for var in zip(self.sdd_mgr.vars,weights ):
            wmc.set_literal_weight(var[0],var[1])
            self.sdd_mgr.auto_gc_and_minimize_off()
            wmc.set_literal_weight(-var[0], 1.0 - var[1])
        w = wmc.propagate()
        return w


#Example lit processor
l1 = [(lambda variables_arr: variables_arr[0]<0), (lambda variables_arr: variables_arr[0]>1), 
        (lambda variables_arr: 0<=variables_arr[0]<=1), (lambda variables_arr: variables_arr[0]==1),
        (lambda variables_arr: variables_arr[0]==variables_arr[1])]
lp1 = Literal_Processor(l1)
lit = lp1.evaluate_literals([1,0])
print(lit)
#Example constraint processor
cp1 = Constraint_Processor([[1,2],[-1,-2]],here,"theory3.cnf") #We assume there is a cnf file in /inputs/simple.cnf
w = [.5,.5]
print(cp1.wmc(w))
