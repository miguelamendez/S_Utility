"""
    Description: This library contains all the functions/classes used for: Matrix generation and modification 
    The library is divided into: Matrix Modifiyers 2d, Matrix Generators 2d, Matrix Modifiyers n-d, Matrix Generators n-d 
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
print("matrices.py:",full_path)
sys.path.append(os.path.dirname(full_path))# two directories above
#print(os.path.dirname(full_path))
from utils.tensors import * #files for imports
#from dir.to.file import *

#External libraries
import numpy as np
import torch


#Matrix Generators:
#All matrix generators are defined using numpy , after creating a matrix you can transform it to a pytorch or tensorflow tensor using the torch.tensor() tensorflow.tensor() functions 
def sp_matrix(m,k):
    """
    :math:``
    Description: Matrix used for the signal perceptron paper
    Implemented:
        [True/False]           
    Args:
        (m:int): the domain size 
        (k:int): the number of variables in the domain

    Return Shape:
        - Input: a tuple of m,k 
        - Output: a matrix of size (m*k,k)

    Examples::

    """
    aix=np.zeros([k]); #Array of indexes (to order them)
    aiw=np.zeros([k]); #Array of indexes (to order them)
    ni=m**k   #Number of Iterations
    n=k  #No. of variables
    nn=m**n #|m^k| domain space
    nnn=m**nn #|Delta|=|m^m^k| function space
    #Matrix
    A=np.zeros([nn,nn],dtype=complex)
    divfrec=m-1
    i=0; j=0
    v=0;
    for xi in range(0,ni,1):
        kx=xi;
        for xj in range(0,k,1):
            aix[xj]= int ( kx % m );
            kx=int(kx/m);
            #print("aix=",aix)
            j=0;
        #First Inner nested loop that generates all combinations of w for a signal
        for wi in range(0,ni,1):
            kw=wi;
            for wj in range(0,k,1): #Generamos los índices
                aiw[wj]= int ( kw % m ) ; #Lo metemos en array
                kw=int(kw/m); #siguientes índices
                #print(i,j,A[i,j],"|",end='')
            exponente=0
            #Seconf Inner loop that  multiplies and sums
            for ii in range(0,k,1):
                exponente=exponente + aix[ii]*aiw[ii]
                exponente=int(exponente)
            #print("exponente=",exponente)
            exponente=1j*np.pi*exponente/divfrec
            #print(exponente)
            #print(np.exp(exponente))
            A[i,j]=np.exp(exponente)
            #print(A[i,j])
            j=j+1
            #print("aiw=",aiw,"j=",j)
            #for aj in range(0,nc,1):
            #	print(i,j,A[i,j],"|",end='')
            #	print()
        i=i+1
    return A

def rsp_matrix(m,k):
    """
    Description: This function creates the matrix used for finding the parameters of  reals signal perceptron using a system of linear equations
    is_Implemented:
        True
    Args:
        (m:int): The domain size , the amount of possible variables that each variable can take
        (k:int): The arity, the amount of variables that each signal can recieve

    Shape:
        - Input: integers that define the functional space
        - Output: a matrix of m

    Examples::
        matrix = rsp_matrix(2,2)
        print(matrix)
        [[0,0,0,0],[0,1,0,0],[0,0,1,0],[1,1]]
    """
    aix=np.zeros([k]); #Array of indexes (to order them)
    aiw=np.zeros([k]); #Array of indexes (to order them)
    ni=m**k   #Number of Iterations
    n=k  #No. of variables
    nn=m**n #|m^k| domain space
    nnn=m**nn #|Delta|=|m^m^k| function space
    # Matrix
    A=np.zeros([nn,nn],dtype=np.float32)
    divfrec=m-1
    i=0; j=0
    v=0;
    for xi in range(0,ni,1):
        kx=xi;
        for xj in range(0,k,1):
            aix[xj]= int ( kx % m );
            kx=int(kx/m);
            #print("aix=",aix)
            j=0;
        #First Inner nested loop that generates all combinations of w for a signal
        for wi in range(0,ni,1):
            kw=wi;
            for wj in range(0,k,1): #Generamos los índices
                aiw[wj]= int ( kw % m ) ; #Lo metemos en array
                kw=int(kw/m); #siguientes índices
                #print(i,j,A[i,j],"|",end='')
            exponente=0
            #Seconf Inner loop that  multiplies and sums
            for ii in range(0,k,1):
                exponente=exponente + aix[ii]*aiw[ii]
                exponente=int(exponente)
            #print("exponente=",exponente)
            exponente=np.pi*exponente/divfrec
            #print(exponente)
            #print(np.exp(exponente))
            A[i,j]=np.cos(exponente)
            #print(A[i,j])
            j=j+1
            #print("aiw=",aiw,"j=",j)
            #for aj in range(0,nc,1):
            #	print(i,j,A[i,j],"|",end='')
            #	print()
        i=i+1
    return A

def walsh_matrix(k):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: 
        - Output: 

    Examples::

    """
    A=np.zeros([nn,nn],dtype=np.float32)
    return A

def rand_matrix(n,m,rng):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: 
        - Output: 

    Examples::
    A=torch.from_numpy(rand_matrix(3,3,2))
    print("matrix",A)

    """
    A=np.random.randint(rng, size=(n, m))
    return A

def gen_matrix(n,m,coord,values=[],tensor=True):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: 
        - Output: 

    Examples::

    """
    if tensor:
        A= torch.zeros(n,m)
        return update_matrix(A,coord,values)
    else:
        A= np.zeros(n,m)
        return update_matrix(A,coord,values)



def freq_gen_sp(m,k):
    """
    Description: This function creates the frecuency array used by the signals of the signal perceptron
    is_Implemented:
        True
    Args:
        (m:int): The domain size , the amount of possible variables that each variable can take
        (k:int): The arity, the amount of variables that each signal can recieve

    Shape:
        - Input: integers that define the functional space
        - Output: an array of size :math:`m*k`

    Examples::
        frequencies = freq_gen_sp(2,2)
        print(frequencies)
        [[0,0],[0,1],[1,0],[1,1]]
           """
    wki=[]
    aiw=np.zeros([k]);
    for i in range(0,m**k,1):
        kw=i;
        for j in range(0,k,1):
            aiw[j]= int ( kw % m );
            kw=int(kw/m);
        w=[]
        for l in aiw:
            w.append(l)
        wki.append(w)
    arrw = np.asarray(wki,dtype=np.float32)
    return arrw


#Matrix Modifiyers
def update_matrix(A,coord,values=[]):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: 
        - Output: 

    Examples::
    Normal:
    A=rand_matrix(3,3,2)
    A=update_matrix(A,[[0,0],[1,1],[2,2]],[3,2,3])
    print(A)
    Tensor Matrix:
    A=torch.from_numpy(rand_matrix(3,3,2))
    print("matrix",A)
    A=update_matrix(A,[[0,0],[1,1],[2,2]],[3,2,3])
    print(A)
    A=rand_matrix(3,3,2)
    print("matrix",A)
    A=update_matrix(A,[[0,0],[1,1],[2,2]],[3,2,3])
    print(A)

    """
    #print("deb_matrix",A)
    #print(coord)
    #print(values)
    if isinstance(A,torch.Tensor):
        x_list=[]
        y_list=[]
        for i in coord:
            x_list.append(i[0])
            y_list.append(i[1])
        coords=[x_list,y_list]
        indices=list2tensors(coords)
        return A.index_put(indices=indices, values=values)
    else:
        for index, crd in enumerate(coord):
            print(index,crd)
            j=crd[0]
            k=crd[1]
            if bool(values):
                A[j][k]=values[index]
            else:
                A[j][k]=1
        return A


def remove_row(A,row_id):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: 
        - Output: 

    Examples::
    A=torch.from_numpy(rand_matrix(3,3,2))
    print("matrix",A)
    A=remove_row(A,1)
    print(A)
    B=rand_matrix(3,3,2)
    print("matrix",B)
    B=remove_row(B,1)
    print(B)
    """
    if isinstance(A,torch.Tensor):
        n=len(A[1])
        index=[]
        for i in range(0,n):
            if i!=row_id:
                index.append(i)
        indices=torch.tensor(index)
        return torch.index_select(A, 0, indices)
    else:
        return np.delete(A, row_id, 0)

def remove_column(A,column_id):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: 
        - Output: 

    Examples::
    A=torch.from_numpy(rand_matrix(3,3,2))
    print("matrix",A)
    A=remove_column(A,1)
    print(A)
    B=rand_matrix(3,3,2)
    print("matrix",B)
    B=remove_column(B,1)
    print(B)
    """
    if isinstance(A,torch.Tensor):
        n = len(A[0])
        index=[]
        for i in range(0,n):
            if i!=column_id:
                index.append(i)
        indices=torch.tensor(index)
        return torch.index_select(A, 1, indices)
    else:
        return np.delete(A, column_id, 1)

def add_row(A,data=[]):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: 
        - Output: 

    Examples::
    A=torch.from_numpy(rand_matrix(3,3,2))
    print("matrix",A)
    A=add_row(A)
    print(A)
    B=rand_matrix(3,3,2)
    print("matrix",B)
    B=add_row(B)
    print(B)
    """
    if not(data):
        n=len(A[0])
        data=np.zeros(n)
    if isinstance(A,torch.Tensor):
        row=torch.tensor(data)
        row=torch.unsqueeze(row, 0)
        return torch.cat((A, row),0)
    else:
        row=data
        return np.vstack([A, row])

def add_column(A,data=[]):
    """
    :math:``
    Description: 
    Implemented:
        [True/False]           
    Args:
        (:): 
        (:): 
            Default:

    Shape:
        - Input: 
        - Output: 

    Examples::
    A=torch.from_numpy(rand_matrix(3,3,2))
    print("matrix",A)
    A=add_column(A)
    print(A)
    B=rand_matrix(3,3,2)
    print("matrix",B)
    B=add_column(B)
    print(B)
    """
    if not(data):
        n=len(A)
        data=np.zeros(n)
    if isinstance(A,torch.Tensor):
        column=torch.tensor(data)
        column=torch.unsqueeze(column, 1)
        return torch.cat((A, column),1)
    else:
        column=data
        return np.column_stack((A,column))
