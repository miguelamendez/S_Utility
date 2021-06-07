"""[This should be a wrapper for the environments currently not implemented]"""
from gym.wrappers import AtariPreprocessing
from semantic_utility.utils.constraint_utils import *
#wrapper should contain this 
next_state,reward,constraint_sat,done
constraint_sat can be obtained by
y_target =self.const_processor.evaluate_literals(variables)
        	const_target=self.const_processor.wmc(y_target)
