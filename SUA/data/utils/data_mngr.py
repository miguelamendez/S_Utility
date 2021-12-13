"""
Description: The data manager is a class for preprocessing and generating information,
"""
import numpy as np
import torch 
#Funcitions for simple data generation

def get_batches_np(data,batch_size):
    #Get batches numpy
    batch=[]
    for i in range(0,int(data.shape[0]), batch_size):
        batch.append(data[i:i+batch_size])
    return batch

def get_batches(data,batch_size):
    #Get batches torch
    batch=torch.tensor([])
    for i in range(0,int(data.size()[0]), batch_size):
        v=torch.unsqueeze(data[i:i+batch_size],0)
        batch=torch.cat((batch,v),0)
    return batch
#Example
#x=torch.randn(10,2)
#print(x)
#print("here",get_batches(x,2))

def rand_idx_np(data,min_data=4):
    #random shuffle numpy
    #Simple dataset shuffler:
    data_size=data.shape[0]
    #Check if there is min data available:
    if data_size >=min_data:
        arr = np.arange(data_size)
        np.random.shuffle(arr)
        return arr
    else:
        return False
#Example:
#x=np.random.randn(20,1,2,2)#20 images of 1*2*2
#idx=rand_idx_np(x)
#print(idx)

def rand_idx(data,min_data=4):
    #Random shuffle torch
    #Simple dataset shuffler:
    data_size=data.shape[0]
    #Check if there is min data available:
    if data_size >=min_data:
        arr=torch.randperm(data_size)
        return arr
    else:
        return False
#Example
#x=torch.randn(10,2)
#print("here",data_randm(x,2))

def seq_data_randm_np(data,seq_len,min_seq=4):
    #This simple function can be used for sample minibatches of sequential data:
    #Examples: RL states , text files
    data_size = data_shape[0]
    if int(data_size/seq_len) >=min_seq:
        arr = np.arange(data_size)
        np.random.shuffle(arr)
        return arr
    else:
        return False
    batch_start = np.arange(0, n_states, self.sequence)
    indices = np.arange(n_states, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i:i+self.batch_size] for i in batch_start]

class MemoryBuffer():
    """A simple memory buffer"""
    def __init__(self,value_size,value_ids=None,mem_size=32):
        #Creating memory list:
        self.data_list=[]
        for i in range (0,value_size):
            self.data_list.append([])
        self.size=mem_size
        self.idx=0
        #Creating Dictionary
        self.ids=value_ids
        if self.ids is not None:
            self.data_dict=dict(zip(value_ids,self.data_list))
        else:
            self.data_dict=None
    
    def __call__(self,key,val_range=None):
        if val_range is None:
            return self.get_values(self,key)
        else:
            vals=self.get_values(self,key)
            return vals[val_range]

    def get_values(self,key):
            if self.data_dict is None:
                raise ValueError("No dictionary exist")
            else:
                return self.data_dict[key]

    def store(self,data):
        if isinstance(data,dict):
                raise ValueError("Not Implemented")
                #Not Implemented
                #if len(self.data_dict[list(data.keys())[0]])+len(list(data.keys())[0])>self.size:
                #    print("Warning: input data overflows memory")
                #    for key in data.keys():
                #        temp_list=self.data_dict[key]+data[key]
                #        self.data_dict[key]=temp_list[-self.mem_size]
                #    else:
        else:
            data_list=data
            if self.idx+len(data_list[0])>self.size:
                print("Warning: input data overflows memory")
                for idx ,sublist in enumerate(self.data_list):
                    temp_list=list(sublist)+data_list[idx]
                    #print(temp_list)
                    self.data_list[idx]=temp_list[-self.size:]
                self.idx=self.size
            else:
                for idx ,sublist in enumerate(self.data_list):
                    self.data_list[idx]=list(sublist)+data_list[idx]
                self.idx=self.idx+len(data_list[0])
        #Convert data list into dict
        if self.ids is not None:
            self.data_dict=dict(zip(self.ids,self.data_list))
    #Deprecated: Working with a single tensor instead of list 
    #def idx_set(self,idx,values=None):
    #        for i,value in enumerate(self.data_list[idx]):
    #            self.data_list[idx][i]=1
    #def set(self,key,values=None):
    #    if self.data_dict is not None:
    #        idx=list(self.data_dict.keys()).index(key)
    #        for i,value in enumerate(self.data_list[idx]):
    #            self.data_list[idx][i]=1
            #Different way to do it:
            #for idx ,value in enumerate(self.data_dict[key]):
            #    self.data_dict[key][idx]=1
    #    else:
    #        raise ValueError("No data dict set for memory")

    def clear(self):
        for idx,value in enumerate(self.data_list):
            self.data_list[idx]=[]
        if self.ids is not None:
            self.data_dict=dict(zip(self.ids,self.data_list))
        self.idx=0
#Example
#ids=["eaea","caca","dada"]
#memory=MemoryBuffer(len(ids),ids,mem_size=8)
#print("List:",memory.data_list)
#print("Dict:",memory.data_dict)
#memory.store([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
#print("List:",memory.data_list)
#print("Dict:",memory.data_dict)
#print("Get %s:%s"%("papa",memory.get_values("papa")))
#print("Mem idx:",memory.idx)
#memory.store([[3,3,3],[3,3,3],[3,3,3]])
#print("List:",memory.data_list)
#print("Dict:",memory.data_dict)
#memory.store([[4],[4],[4]])
#print("List:",memory.data_list)
#print("Dict:",memory.data_dict)
#memory.clear()
#print("List:",memory.data_list)
#print("Dict:",memory.data_dict)

class DataManager():
    """A simple data manager, it generates the batches over the current memory"""
    """Will get the data and some information of how"""
    def __init__(self,memory_dict,mngr_fn,pre_proc_fn=None,batch_size=4):
        self.memory=memory_dict
        self.model=mngr_fn
        self.preprocess=pre_proc_fn
        self.batch_size=batch_size #Currently this is fixed but can be defined by the self.model outputs when optimizing
    
    def __call__(self,data_type):
        return self.process(data_type)
        
    def process(self,data_type):
        """To Do: In the future self.model could do those two steps by itself (get the data and generate the batches)"""
        #Get the indexes of the data to be used: idx.size() <= data.size()[0]
        idx = self.model(self.memory[data_type])
        data =self.memory[data_type][idx]
        #Generate batches
        batches=get_batches(data,self.batch_size)
        return batches 
        
    def store(self,data_type,data):
        if self.preprocess is not None:
            self.memory[data_type].append(self.preprocess(data))
        else:
            self.memory[data_type].append(data)

    def clear(self):
        for key in self.memory:
            self.memory[key]=[]

#Example:
#x_t =torch.randn(20,1,2,2)
#print("x",x_t)
#memory={"images": x_t} #saving as batch
#manager_func=rand_idx #generating model function
#data_mngr=DataManager(memory,manager_func) #building data mngr
#data=data_mngr("images") #obtaining data
#for idx,batch in enumerate(data):#Printing results
#    print("Batch %s: %s"%(idx,batch))

def build():
    return 

list=[]
