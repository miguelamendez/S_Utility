import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from signal_perceptron import *
from sp_paper_models import *
import time
from train import *



#Model Class:
import torch

#Loading datasets:
#######################################################################################################################

training_data_mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data_mnist = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)



train_mnist_dataloader = DataLoader(training_data_mnist, batch_size=batch_size)
test_mnist_dataloader = DataLoader(test_data_mnist, batch_size=batch_size)




    
def main():
    training_data_mnist = datasets.MNIST(root="data",train=True,download=True,transform=ToTensor(),)
    test_data_mnist = datasets.MNIST(root="data",train=False,download=True,transform=ToTensor(),)
    train_mnist_dataloader = DataLoader(training_data_mnist, batch_size=batch_size)
    test_mnist_dataloader = DataLoader(test_data_mnist, batch_size=batch_size)
    fsp=Model()
    fsp.optimizer=
    fsp.loss_fn=
    fsp.device=
    loss=fsp.train(train_mnist_dataloader)
    fsp.test()


def train(data)
    train_features, train_labels = next(iter(data))
def learntrain(train_mnist_dataloader,test_mnist_dataloader,train_f_mnist_dataloader,test_f_mnist_dataloader):
    #Loading Models:
    ##################################################################################################################################


    arch = FSP_mnist(128).to(device)
    
    fsp= Model(arch,optimizer=,loss=,PATH="data/models/idm_FSP128_mnist.pt")

    #MODEL PROPERTIES:
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

    #Printing Learnable Parameters
    ################################################################################################################################
    fsp128_parameters = filter(lambda p: p.requires_grad, fsp128.parameters())
    fsp128_params = sum([np.prod(p.size()) for p in fsp128_parameters])
    fsp_parameters = filter(lambda p: p.requires_grad, fsp.parameters())
    fsp_params = sum([np.prod(p.size()) for p in fsp_parameters])
    mlp1_parameters = filter(lambda p: p.requires_grad, mlp1.parameters())
    mlp1_params = sum([np.prod(p.size()) for p in mlp1_parameters])
    mlp2_parameters = filter(lambda p: p.requires_grad, mlp2.parameters())
    mlp2_params = sum([np.prod(p.size()) for p in mlp2_parameters])

    print("Learnable Parameters for MNIST models:")
    print("FSP128 \t FSP512 \t MLP 1 hidden  \t MLP 2 hidden")
    print(fsp128_params,"\t",fsp_params,"\t",mlp1_params,"\t",mlp2_params)

    #################################################################################################################################
    #Memory:


    #Forward PassTime
    #################################################################################################################################

    train_features, train_labels = next(iter(train_mnist_dataloader))
    inputs=train_features[0]
    inputs=inputs.to(device)
    print(train_features.size())
    print(inputs.size())
    #Warmup
    t2 = time.time()
    pred2=mlp1(inputs)
    elapsed2 = time.time() - t2
    timer2=Timer(mlp1,inputs)
    ############
    t22 = time.time()
    pred22=mlp1(inputs)
    elapsed22 = time.time() - t22
    timer22=Timer(mlp1,inputs)
    t3 = time.time()
    pred3=mlp2(inputs) 
    elapsed3 = time.time() - t3
    timer3=Timer(mlp2,inputs)
    t11 = time.time()
    pred1=fsp128(inputs)
    elapsed11 = time.time() - t11
    timer11=Timer(fsp128,inputs)
    t1 = time.time()
    pred1=fsp(inputs)
    elapsed1 = time.time() - t1
    timer1=Timer(fsp,inputs)
    print("Forward time for MNIST models:")
    print("FSP128 \t FSP512 \t MLP 1 hidden  \t MLP 2 hidden")
    print(elapsed11,"\t",elapsed1,"\t",elapsed22,"\t",elapsed3)
    print("Forward time for MNIST models FS Timer class:")
    print("FSP128 \t FSP512 \t MLP 1 hidden  \t MLP 2 hidden")
    print(timer11.mean(),"\t",timer1.mean(),"\t",timer22.mean(),"\t",timer3.mean())
    #Profiler(Only Pytorch)------------------------------------


    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference mlp1"):
            mlp1(inputs)
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference mlp1"):
            mlp1(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference mlp2"):
            mlp2(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference fsp128"):
            fsp128(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference fsp512"):
            fsp(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #################################################################################################################################
    #Backward PassTime
    #################################################################################################################################

    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################



    #MODELS TRAINING
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #Training Hyperparameters:
    #----------------------------------------------------------
    epochs = 9
    lr=.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fsp128.parameters(), lr=lr)
    optimizer1 = torch.optim.Adam(fsp.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(mlp1.parameters(), lr=lr)
    optimizer3 = torch.optim.Adam(mlp2.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(fsp128.parameters(), lr=lr)
    #optimizer1 = torch.optim.SGD(fsp.parameters(), lr=lr)
    #optimizer2 = torch.optim.SGD(mlp1.parameters(), lr=lr)
    #optimizer3 = torch.optim.SGD(mlp2.parameters(), lr=lr)
    print("Optimizer: Adam ,lr:" ,lr)
    #print("Optimizer: SGD ,lr:" ,lr)
    #----------------------------------------------------------

    print("Training models with MNIST DATASET :")
    print("Fourier Signal Perceptron 128")
    optimal_epoch=[]
    time_backprop=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss1,tb=train_mnist(train_mnist_dataloader, fsp128, loss_fn, optimizer,device)
        accuracy,loss2=test_mnist(test_mnist_dataloader, fsp128, loss_fn,device)
        if not bool(optimal_epoch):
            optimal_epoch=[t,accuracy, loss2,loss1]
        if bool(optimal_epoch):
            if optimal_epoch[2]>loss2:
                optimal_epoch=[t,accuracy, loss2,loss1]
        time_backprop.append(tb)
    print("Backprop time:")
    con=np.concatenate(time_backprop)
    print(np.mean(con))
    print("Final  epoch:")
    print(epochs,accuracy,loss2,loss1)
    print("Optimal  epoch:")
    print(optimal_epoch)

    print("Fourier Signal Perceptron 512")
    optimal_epoch=[]
    time_backprop=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss1,tb=train_mnist(train_mnist_dataloader, fsp, loss_fn, optimizer1,device)
        accuracy,loss2=test_mnist(test_mnist_dataloader, fsp, loss_fn,device)
        if not bool(optimal_epoch):
            optimal_epoch=[t,accuracy, loss2]
        if bool(optimal_epoch):
            if optimal_epoch[2]>loss2:
                optimal_epoch=[t,accuracy, loss2]
    time_backprop.append(tb)
    print("Backprop time:")
    con=np.concatenate(time_backprop)
    print(np.mean(con))
    print("Final  epoch:")
    print(epochs,accuracy,loss2,loss1)
    print("Optimal  epoch:")
    print(optimal_epoch)

    print("MLP 1 hidden layer Signal Perceptron")
    optimal_epoch=[]
    time_backprop=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss1,tb=train_mnist(train_mnist_dataloader, mlp1, loss_fn, optimizer2,device)
        accuracy,loss2=test_mnist(test_mnist_dataloader, mlp1, loss_fn,device)
        if not bool(optimal_epoch):
            optimal_epoch=[t,accuracy, loss2,loss1]
        if bool(optimal_epoch):
            if optimal_epoch[2]>loss2:
                optimal_epoch=[t,accuracy, loss2,loss1]
        time_backprop.append(tb)
    print("Backprop time:")
    con=np.concatenate(time_backprop)
    print(np.mean(con))
    print("Final  epoch:")
    print(epochs,accuracy,loss2,loss1)
    print("Optimal  epoch:")
    print(optimal_epoch)

    print("MLP 2 hidden layer Signal Perceptron")
    optimal_epoch=[]
    time_backprop=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss1,tb=train_mnist(train_mnist_dataloader, mlp2, loss_fn, optimizer3,device)
        accuracy,loss2=test_mnist(test_mnist_dataloader, mlp2, loss_fn,device)
        if not bool(optimal_epoch):
            optimal_epoch=[t,accuracy, loss2,loss1]
        if bool(optimal_epoch):
            if optimal_epoch[2]>loss2:
                optimal_epoch=[t,accuracy, loss2,loss1]
        time_backprop.append(tb)
    print("Backprop time:")
    con=np.concatenate(time_backprop)
    print(np.mean(con))
    print("Final  epoch:")
    print(epochs,accuracy,loss2,loss1)
    print("Optimal  epoch:")
    print(optimal_epoch)

    print("Training models with FashionMNIST DATASET :")
    #Loading initial parameters
    fsp128.load_state_dict(torch.load(PATH))
    fsp.load_state_dict(torch.load(PATH1))
    mlp1.load_state_dict(torch.load(PATH2))
    mlp2.load_state_dict(torch.load(PATH3))
    epochs = 9
    print("Fourier Signal Perceptron 128")
    optimal_epoch=[]
    time_backprop=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss1,tb=train_mnist(train_f_mnist_dataloader, fsp128, loss_fn, optimizer,device)
        accuracy,loss2=test_mnist(test_f_mnist_dataloader, fsp128, loss_fn,device)
        if not bool(optimal_epoch):
            optimal_epoch=[t,accuracy, loss2,loss1]
        if bool(optimal_epoch):
            if optimal_epoch[2]>loss2:
                optimal_epoch=[t,accuracy, loss2,loss1]
        time_backprop.append(tb)
    print("Backprop time:")
    con=np.concatenate(time_backprop)
    print(np.mean(con))
    print("Final  epoch:")
    print(epochs,accuracy,loss2,loss1)
    print("Optimal  epoch:")
    print(optimal_epoch)

    print("Fourier Signal Perceptron 512")
    optimal_epoch=[]
    time_backprop=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss1,tb=train_mnist(train_f_mnist_dataloader, fsp, loss_fn, optimizer1,device)
        accuracy,loss2=test_mnist(test_f_mnist_dataloader, fsp, loss_fn,device)
        if not bool(optimal_epoch):
            optimal_epoch=[t,accuracy, loss2,loss1]
        if bool(optimal_epoch):
            if optimal_epoch[2]>loss2:
                optimal_epoch=[t,accuracy, loss2,loss1]
        time_backprop.append(tb)
    print("Backprop time:")
    con=np.concatenate(time_backprop)
    print(np.mean(con))
    print("Final  epoch:")
    print(epochs,accuracy,loss2,loss1)
    print("Optimal  epoch:")
    print(optimal_epoch)

    print("MLP 1 hidden layer")
    optimal_epoch=[]
    time_backprop=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss1,tb=train_mnist(train_f_mnist_dataloader, mlp1, loss_fn, optimizer2,device)
        accuracy,loss2=test_mnist(test_f_mnist_dataloader, mlp1, loss_fn,device)
        if not bool(optimal_epoch):
            optimal_epoch=[t,accuracy, loss2,loss1]
        if bool(optimal_epoch):
            if optimal_epoch[2]>loss2:
                optimal_epoch=[t,accuracy, loss2,loss1]
        time_backprop.append(tb)
    print("Backprop time:")
    con=np.concatenate(time_backprop)
    print(np.mean(con))
    print("Final  epoch:")
    print(epochs,accuracy,loss2,loss1)
    print("Optimal  epoch:")
    print(optimal_epoch)

    print("MLP 2 hidden layer")
    optimal_epoch=[]
    time_backprop=[]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss1,tb=train_mnist(train_f_mnist_dataloader, mlp2, loss_fn, optimizer3,device)
        accuracy,loss2=test_mnist(test_f_mnist_dataloader, mlp2, loss_fn,device)
        if not bool(optimal_epoch):
            optimal_epoch=[t,accuracy, loss2,loss1]
        if bool(optimal_epoch):
            if optimal_epoch[2]>loss2:
                optimal_epoch=[t,accuracy, loss2,loss1]
        time_backprop.append(tb)
    print("Backprop time:")
    con=np.concatenate(time_backprop)
    print(np.mean(con))
    print("Final  epoch:")
    print(epochs,accuracy,loss2,loss1)
    print("Optimal  epoch:")
    print(optimal_epoch)

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
import sys
for i in range(5):
    orig_stdout = sys.stdout
    subname=str(i)+"_"
    out="data/experiments/exp2/run1/"+subname+".txt"
    f = open(out, 'w')
    sys.stdout = f
    full_analysis_train(train_mnist_dataloader,test_mnist_dataloader,train_f_mnist_dataloader,test_f_mnist_dataloader)             
    sys.stdout = orig_stdout
    f.close()
