"""CNF_utils process numpy lists and .cnf DIMACS files """
import pickle
import numpy as np

class CNF_Utils():
    def dimacs_cnf_to_list(self,file_name_path="test.cnf"):
        start = False
        cnf_formula = []
        with open(file_name_path) as f:
            for line in f:
                if line.startswith('p cnf'):
                    line = line.strip()
                    nums = line.split()
                    num_clauses = int(nums[3])
                    num_vars = int(nums[2])
                    start = True
                    continue
                elif line.startswith('%'):
                    break
                if not start: continue
                line = line.strip()
                nums = line.split()
                nums = nums[:-1] # get rid of 0
                nums = [int(n) for n in nums]
                cnf_formula.append(nums)
        return cnf_formula , num_clauses , num_vars

    def list_to_dimacs_cnf(self,cnf_formula,file_name_path="test.cnf"):
        flat_list = []
        for sublist in cnf_formula:
            for item in sublist:
                flat_list.append(item)
        _vars=[abs(ele) for ele in flat_list]
        _vars= np.unique(_vars)
        num_vars=len(_vars)
        num_clauses = len(cnf_formula)
        f = open(file_name_path,"w")
        f.write("c cnf file created from python list \n")
        pline= 'p cnf '+str(num_vars)+'  '+str(num_clauses)+'\n'
        f.write(pline)
        for i in cnf_formula:
            clause=""
            for j in i :
                print()
                clause =clause +str(j)+" "
            clause = clause +"0 \n"
            f.write(clause)
        return num_vars, num_clauses

    def cnf_to_pickle(self,cnf_formula,file_name="test_cnf.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(cnf_formula, f)

    def dimacs_cnf_to_pickle(self,file_name_dimacs_path,file_name_path="test_cnf.pkl"):
        cnf_formula ,num_vars, num_clauses = self.dimacs_cnf_to_list(file_name_dimacs_path)
        with open(file_name_path, 'wb') as f:
            pickle.dump(cnf_formula, f)

    def pickle_to_list(self,file_name_path="test_cnf.pkl"):
        with open(file_name_path, 'rb') as f:
            cnf_formula = pickle.load(f)
        return cnf_formula
    
#Examples:
#cnf_util = CNF_Utils()
#print(cnf_util.list_to_dimacs_cnf([[1,-2],[2,3,4],[2,4],[6,-1]]))
#print(cnf_util.dimacs_cnf_to_list())
