import numpy as np
import pickle

def gen_finite_array(m,length,size):
    yki=[]
    aiy=np.zeros([length]);
    for i in range(0,size,1):
        ky=i;
        for j in range(0,length,1):
            aiy[j]= int ( ky % m );  
            ky=int(ky/m); 
        yt=[]
        for l in aiy:
            yt.append(l)
        yki.append(yt)
    y = np.asarray(yki)
    return y


def data_gen(m,k,func_samples=[]):
    if bool(func_samples):
        #Creating n random functions
        Y=np.random.randint(m, size=(func_samples,m**k))
    else:
        #Creating all possible functions 
        Y=gen_finite_array(m,length=m**k,size=m**(m**k))
    #Creating Dataset that is all possible combinations of the inputs
    X=gen_finite_array(m,length=k,size=(m**k))
    return X, Y

def partial_data_gen():
    
    return train_X,train_Y,test_X,test_Y

