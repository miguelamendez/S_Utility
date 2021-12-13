"""
Description: This library contains all the functions/classes for: process numpy lists and .cnf DIMACS files as well as some generators.
    Please refer to each of the functions/classes for a full description of what they do.
    Functions ([first,second] "first bool" notes the implemented functions "second bool" notes the implemented documentation):
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary
        name:[True,False] summary


    Classes ([first,second] "first bool" notes the implemented classes "second bool" notes the implemented documentation):
        CNF_transforms:[True,False] This class contains a list of functions to transform lists into cnf files into pickle files
        CNF_generator:[True,False] This class contains a list of functions that generate cnf files.
        name:[True,False] summary
        name:[True,False] summary
        """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("cnf.py:",full_path)
sys.path.append(os.path.dirname(full_path))# one directory above
#from dir.to.file import * #files for imports
#from dir.to.file import *

#External libraries
import pickle
import numpy as np
from  cnfgen import RandomKCNF as randkcnf

class CNF_processing():
    """Description: The processing class takes care on transformations on a CNF constraint, this class is made under the assumption that any cnf can be reduced into a 3-sat function. 
        Saying that, all functions defined in this class will return a processed CNF constraint in the form of 3-sat CNF or smaller."""
    
    def __init__(self,cnf_list=[]):
        self.cnf = cnf_list
        self.sat3 = []
        self.sat2 = []
        self.sat1 = []
        self.order_cnf()
    
    def set_cnf(self,cnf_list):
        self.cnf = cnf_list
        self.order_cnf()

        
    def order2Terms(self,t,i,j):
        if (abs(t[i]) > abs(t[j])):
            t[i],t[j]=t[j],t[i]
        return t

    def order3Terms(self,t):
        t=self.order2Terms(t,0,1)   
        t=self.order2Terms(t,1,2)
        t=self.order2Terms(t,0,1)
        return t
    
    def sort1(self,l):
        l=np.squeeze(l,axis=1)
        l=list(dict.fromkeys(l))
        return np.expand_dims(l, axis=1)

    def sort2(self,l):
        for i in range(0,len(l)):
            l[i]=self.order2Terms(l[i],0,1)
        return l 

    def sort3(self,l):
        for i in range(0,len(l)):
            l[i]=self.order3Terms(l[i])
        return l

    def formatsat(self,cnf,sat):
        l=cnf
        if sat==1:
            self.sort1(l)
        if sat==2:
            self.sort2(l)
        if sat==3:
            self.sort3(l)
        #print(l)    
        gl=[]
        for i in l:
            if i not in gl:
                gl.append(i)
        return gl

    def get_sat(self):
            for i in self.cnf:
                if len(i)==3:
                    self.sat3.append(i)
                elif len(i)==2:
                    self.sat2.append(i)
                elif len(i)==1:
                    self.sat1.append(i)

    def order_cnf(self):
        self.get_sat()
        #print(self.sat3,self.sat2,self.sat1)
        if self.sat3!=[]:
            self.sat3=self.formatsat(self.sat3,sat=3)
        if self.sat2!=[]:
            self.sat2=self.formatsat(self.sat2,sat=2)
        if self.sat1!=[]:
            self.sat1=self.formatsat(self.sat1,sat=1)
        self.cnf=self.sat3+self.sat2+self.sat1
                    
        
                    
                
#Example CNF_processing
#theory=CNF_processing()
#theory.set_cnf([[3,1,1],[3,1,1],[1],[2],[1],[2,1],[2,1],[1,2],[-2,1],[-1,2]])
#print(theory.cnf)

class CNF_formats():
    def dimacs_to_list(self,file_name_path="test.cnf"):
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
        return {"cnf":cnf_formula , "clauses":num_clauses , "vars":num_vars}

    def list_to_dimacs(self,cnf_formula,file_name_path="test.cnf"):
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
        return {"clauses":num_clauses , "vars":num_vars}

    def cnf_to_pickle(self,cnf_formula,file_name="test_cnf.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(cnf_formula, f)

    def dimacs_to_pickle(self,file_name_dimacs_path,file_name_path="test_cnf.pkl"):
        cnf_formula ,num_vars, num_clauses = self.dimacs_cnf_to_list(file_name_dimacs_path)
        with open(file_name_path, 'wb') as f:
            pickle.dump(cnf_formula, f)

    def pickle_to_list(self,file_name_path="test_cnf.pkl"):
        with open(file_name_path, 'rb') as f:
            cnf_formula = pickle.load(f)
        return cnf_formula
#Example CNF_formats
#procss=CNF_processing([[3,1,1],[3,1,1],[1],[2],[1],[2,1],[2,1],[1,2],[-2,1],[-1,2]])
#formats=CNF_formats()
#data=formats.list_to_dimacs(procss.cnf,"test1.cnf")
#print(data)
#print(formats.dimacs_to_list("test1.cnf"))

class CNF_generators():
    
    def cnf_rand(self,n,m,sat=3):
        if n<sat:
            raise ValueError("The number of variables cannot be smaller than the amount of variables per clause")
        """This funciton generates random constraints in 3 sat, this could be unSAT
        n: number of variables, should be bigger than sat
        m:number of clauses
        """
        cnf_list=[]
        while len(cnf_list)<m:
            clause=[]
            while len(clause)<sat:
                x=np.random.randint(1,n+1)
                if x not in clause:
                    clause.append(x)
            for index,value in enumerate(clause):
                sign=np.random.randint(2)
                if sign==1:
                    clause[index]=-value
            if clause not in cnf_list:
                cnf_list.append(clause)
        return cnf_list

    def cnf_rand_sat(self,n,m):
        gen=randkcnf(3, n, m)
        print(gen.is_satisfiable())
        return gen

    def cnf_rand_unsat(self,n,m):
        return

gen=CNF_generators()
print(gen.cnf_rand_sat(3,5))
