from pathlib import Path
import math
from pysdd.sdd import SddManager, Vtree, WmcManager
from semantic_utility.utils.cnf_utils import *
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
    def __init__(self, formula,path,formula_name="test_formula.cnf"):
        """[summary]
            formula = Propositional logic constraint a.k.a Theory 
            formula_name =name of the propositional formula Example theory1.cnf
            variables_arr = a list of variables observed by the agent 
            literal_arr = is an array of functions usually lambdas defined over the variables_arr
            Example:l1 = [(lambda variables_arr: variables_arr[0]<0),(lambda variables_arr: variables_arr[0]>1),(lambda variables_arr: 0<=variables_arr[0]<=1),(lambda variables_arr: variables_arr[0]==1),(lambda variables_arr: variables_arr[0]==variables_arr[1])]
        """
        print(path)
        self.cnf_prcss =CNF_Utils()
        if len(formula)!=0:
            formula_path="inputs/"+formula_name
            print(formula_path)
            #self.formula = self.formula_to_cnf_list(formula) #Needs to be implemented 
            self.formula = formula
            num_vars , num_clauses = self.cnf_prcss.list_to_dimacs_cnf(self.formula,formula_path) 
            self.sdd_mgr, self.sdd_node= SddManager.from_cnf_file(bytes(path / "inputs" / formula_name))
        else:
            self.sdd_mgr, self.sdd_node= SddManager.from_cnf_file(bytes(path / "inputs" / "base_theory.cnf"))
        name= formula_name.split(".")
        sdd_name=str(path)+"/outputs/SDD/sdd_"+name[0]+".dot"
        with open(sdd_name, "w") as out:
            print(self.sdd_mgr.dot(), file=out)

    def formula_to_cnf_list(self,formula):
        """Transform a propositional constraint into a list of clauses in cnf :Needs to be implemented"""
        pass


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
            print (var)
            wmc.set_literal_weight(var[0],var[1])
            self.sdd_mgr.auto_gc_and_minimize_off()
            wmc.set_literal_weight(-var[0], 1.0 - var[1])
        w = wmc.propagate()
        return w

