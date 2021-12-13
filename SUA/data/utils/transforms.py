"""This utilities are for data transformation"""
import numpy as np
import torch

def batchify(data: Tensor, bsz: int,clean=True) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    if clean:
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device)
    else:
        data.size(0)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)




#NUMPY FUNCTIONS
def int_to_bin(arr,num_bits=8):
    """This function transforms an array of integers into an equivalent array of binary numbers
    Inputs:
    arr [type: arr]. Description: arr is the array to be processed
    num_bits [type: int]. Description: numbits is the size of the bit array where each of the integers are going to be stored. Default:8
    """
    temp_arr=[]
    for i in range(0,len(arr)):
        f="0"+str(num_bits)+"b"
        a=format(arr[i],f)
        b=np.zeros(num_bits)
        for element in range(0, len(a)):
            b[element]=float(a[element])
        temp_arr.append(b)
    bin_arr=np.asarray(temp_arr)
    return bin_arr

def int_to_bin_dataset(dataset,num_bits=8):
    """This function transforms a complete dataset of integers into an equivalent array of binary numbers
    Inputs:
    arr [type: arr]. Description: arr is the array to be processed
    num_bits [type: int]. Description: numbits is the size of the bit array where each of the integers are going to be stored. Default:8
    """
    bin_train_list=[]
    for data in dataset:
        k=0
        bin_data_list=[]
        for element in data:
            bin_element=int_to_bin(element)
            k+=1
            bin_data_list.append(bin_element)
        bin_data_array=np.asarray(bin_data_list)
        bin_train_list.append(bin_data_array)
    bin_train_array=np.stack(bin_train_list)
    return bin_train_array

#PYTORCH FUNCTIONS
