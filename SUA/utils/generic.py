"""
    Description: This library contains all the functions for transforming lists and dictionaries and other python related structure transformations
    Please refer to each of the functions/classes for a full description of what they do.

    Functions ([first,second] "first bool" notes the implemented functions "second bool" notes the implemented documentation):
        list2dict:[True,False] Transforms a "list" into a dictionary where the keys are integers ranging from 0 to len("list")
        keys2list:[True,False] Creates a list using the keys of the dictionary
        vals2list:[True,False] Creates a list using the values of the dictionary
        """
#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("generic.py:",full_path)
sys.path.append(os.path.dirname(full_path))# one directory above
#print(os.path.dirname(full_path))
#from dir.to.file import * #files for imports
#from dir.to.file import *

#External libraries
def list2dict(some_list,keys=False):
    #Example:
    #node_list=["s","sd"]
    #print(list2dict(node_list,keys=True))
    if keys:
        return(dict(list([values,idx]for idx,values in enumerate(some_list))))
    else:
        return(dict(list([idx,values]for idx,values in enumerate(some_list))))

    
def keys2list(some_dict):
    #Example:
    #node_dict={1:"s",2:"sd"}
    #print(keys2list(node_dict))
    return list(some_dict.keys())

def vals2list(some_dict):
    #Example:
    #node_dict={1:"s",2:"sd"}
    #print(vals2list(node_dict))
    return list(some_dict.values())

def tlist2list(some_tuple_list):
    some_list=[]
    for i in some_tuple_list:
        some_list.append(list(i))
    return some_list

def build_1val_dict(keys,value):
    new_dict={}
    for i in keys:
        new_dict[i]=value
    return new_dict
