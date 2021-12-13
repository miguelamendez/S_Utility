from semantic_utility.utils.constraint_utils import * 

here = Path(__file__).parent

l1 = [(lambda variables_arr: variables_arr[0]<0),(lambda variables_arr: variables_arr[0]>1),\
        (lambda variables_arr: 0<=variables_arr[0]<=1),(lambda variables_arr: variables_arr[0]==1),\
        (lambda variables_arr: variables_arr[0]==variables_arr[1])
     ]
cp1 = Constraint_Processor([[1,2],[-1,-2]],here,"theory3.cnf") #We assume there is a cnf file in /inputs/simple.cnf
lp1 = Literal_Processor(l1)

lit=lp1.evaluate_literals([1,0])
print(lit)
w=[.5,.5]
print(cp1.wmc(w))
