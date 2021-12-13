"""
    Description: This library contains all the functions/classes for: training models and agents
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
print("train_fn.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path))) # one directory above


#External libraries
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import time

def dynamic_train(agent,env,ext_data_mngr=None,pre_proc_fn=None,episodes=16):
    #This is the high-level function for training agents:
    #Cheking valid attributes:
    if len(pre_proc_fn)>1:
        raise ValueError("Preprocessing functions for multiple state_spaces not supported")
    else:
        pre_proc=pre_proc_fn["space_0"]
    if ext_data_mngr==None:
        print("Training will be done using internal agent data_mngr")
        if agent.data_mngr==None:
            raise ValueError("Agent cannot be trained:Not internal or external data manager")
    #Initializing external training dictionary
    train_dict={}
    #Main train loop:
    for episode in range(0,episodes):
        state=env.reset()
        state=pre_proc(state) if pre_proc is not None else state #Preprocessing initial state
        agent.start(torch.tensor(state)) #Initializing agent (some agents require some initial values to start doing training and inference also it resets the internal memory)
        done = False
        score = 0
        n_steps = 0
        episode_data=[]
        while not done:
            #Choose Action-----------------------------------------------------------------------
            action = agent.act(torch.tensor(state)) #Agent acts
            next_state, reward, done, info = env.step(action) #Environment step 
            next_state = pre_proc(next_state) if pre_proc is not None else next_state #Preprocessing next state if there is a preprocessing function
            data_dict={"next_state":torch.tensor(next_state), "rewards":torch.tensor(reward), "done":torch.tensor(done)} #Creating data_dict
            ext_data_mngr.store(data_dict) if ext_data_mngr is not None else agent.store(data_dict)#Storing data_dict
            learned_data=agent.learn(ext_data_mngr())
            agent.process(data_dict) #Agent processing function
            n_steps += 1
            score += reward
            episode_data.append([n_steps,score])
        train_dict[episode]=episode_data
    env.close()
    return train_dict

"""TRAINING using Data_Loaders"""
def static(data_mngr,model):
    size = len(data_mngr)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        #start=time.time()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_hist.append(loss.item())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #end = time.time()-start
        #time_backward.append(end)
        #tb=np.asarray(time_backward)
    return loss

"""Some dummy learning"""
def dummy_train(data, arch, obj_fn, optimizer,device):
    x,y=data
    #print("here",device)
    #X,Y = x.to(device), y.to(device)
    print(x,y)
    loss_hist=[]
    pred =arch(x)
    print("pred",pred)
    loss=obj_fn(pred,y)
    optimizer.zero_grad()
    loss_hist.append(loss.item())
    loss.backward()
    optimizer.step()
    return loss

#Train loops for first set of experiments (check exp1.py)
def train_pytorch(x_train,y_train,model,PATH,epochs,optimizer,loss_fn):
    total_hist=[]
    final_loss=[]
    learned_epochs=[]
    total_time=[]
    for i in y_train:
        model.load_state_dict(torch.load(PATH))
        i = i.unsqueeze(1)
        history_train=[]
        learned_epoch=[]
        time_backward=np.zeros(epochs)
        for j in range(0,epochs):
            start=time.time()
            pred=model(x_train)
            loss = loss_fn(pred,i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time.time()-start
            time_backward[j]=end
            history_train.append([j,loss])
            if not bool(learned_epoch):
                if loss<=.001:
                    learned_epoch.append(j)
        learned_epochs.append(learned_epoch)
        final_loss.append(loss.detach().numpy())
        total_hist.append(history_train)
        total_time.append(time_backward)
    l=0
    for i in final_loss:
        l=l+i
    avg_fl=l/len(final_loss)
    return total_hist,avg_fl,learned_epochs,total_time

#Train loops for second set of experiments (check exp2.py)
def train_mh_pytorch(x_train,y_train,model,PATH,epochs,optimizer,loss_fn):
    y_train=torch.transpose(y_train, 0, 1)
    total_hist=[]
    final_loss=[]
    learned_epoch=[]
    model.load_state_dict(torch.load(PATH))
    history_train=[]
    time_backward=np.zeros(epochs)
    for j in range(0,epochs):
        start=time.time()
        pred=model(x_train)
        loss = loss_fn(pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()-start
        time_backward[j]=end
        history_train.append([j,loss.detach().numpy()])
        if not bool(learned_epoch):
            if loss<=.001:
                learned_epoch.append(j)
    final_loss=loss.detach().numpy()
    total_hist.append(history_train)
    return total_hist,final_loss,learned_epoch,time_backward

"""MNIST TRAINING LOOPS"""
def train_mnist(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    time_backward=[]
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        start=time.time()
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()-start
        time_backward.append(end)
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        tb=np.asarray(time_backward)
    return loss ,tb

def test_mnist(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct) ,test_loss
    
    
    
"""
    Description: This library contains all the functions/classes for: training with dynamic data (e.g reinforcement learning environments)
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
print("dynamic.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path))) # one directory above


#External libraries
import torch
import numpy as np

def dynamic(agent,env,ext_data_mngr=None,pre_proc_fn=None,episodes=16,test_mode=False):
    #This is the high-level function for training agents:
    #Cheking valid attributes:
    if len(pre_proc_fn)>1:
        raise ValueError("Preprocessing functions for multiple state_spaces not supported")
    else:
        pre_proc=pre_proc_fn["space_0"]
    if ext_data_mngr==None:
        print("Training will be done using internal agent data_mngr")
        if agent.data_mngr==None:
            raise ValueError("Agent cannot be trained:Not internal or external data manager")
    #Initializing external training dictionary
    train_dict={}
    #Main train loop:
    for episode in range(0,episodes):
        state=env.reset()
        state=pre_proc(state) if pre_proc is not None else state #Preprocessing initial state
        agent.start(torch.tensor(state)) #Initializing agent (some agents require some initial values to start doing training and inference also it resets the internal memory)
        done = False
        score = 0
        n_steps = 0
        episode_data=[]
        while not done:
            #Choose Action-----------------------------------------------------------------------
            action = agent.act(torch.tensor(state)) #Agent acts
            next_state, reward, done, info = env.step(action) #Environment step 
            next_state = pre_proc(next_state) if pre_proc is not None else next_state #Preprocessing next state if there is a preprocessing function
            data_dict={"next_state":torch.tensor(next_state), "rewards":torch.tensor(reward), "done":torch.tensor(done)} #Creating data_dict
            ext_data_mngr.store(data_dict) if ext_data_mngr is not None else agent.store(data_dict)#Storing data_dict
            learned_data=agent.learn(ext_data_mngr())
            agent.process(data_dict) #Agent processing function
            n_steps += 1
            score += reward
            episode_data.append([n_steps,score])
        train_dict[episode]=episode_data
    env.close()
    return train_dict

def test(agent,env,pre_proc_fn=None,episodes=4,render=False):
    #Cheking valid attributes:
    if len(pre_proc_fn)>1:
        raise ValueError("preprocessing functions for multiple state_spaces not supported")
    else:
        pre_proc=pre_proc_fn["space_0"]
    #Initializing external training dictionary
    test_dict={}
    #Main train loop:
        #Main train loop:
    for episode in range(0,episodes):
        state=env.reset()
        state=pre_proc(state) if pre_proc is not None else state #Preprocessing initial state
        agent.start(torch.tensor(state)) #Initializing agent (some agents require some initial values to start doing training and inference also it resets the internal memory)
        done = False
        score = 0
        n_steps = 0
        episode_data=[]
        while not done:
            if render:
                env.render()
            #Choose Action-----------------------------------------------------------------------
            action = agent.act(torch.tensor(state))
            next_state, reward, done, info = env.step(action)
            next_state = pre_proc(next_state) if pre_proc is not None else next_state
            data_dict={"next_state":torch.tensor(next_state), "reward":torch.tensor(reward), "done":torch.tensor(done)}
            agent.process(data_dict) #agent.process()
            n_steps += 1
            score += reward
            episode_data.append([n_steps,score])
        test_dict[episode]=episode_data
    env.close()
    return test_dict
    
import torch

import matplotlib.pyplot as plt
import time



#Model Class:
import torch

#Loading datasets:
#######################################################################################################################


    
def main():
    training_data_mnist = datasets.MNIST(root="data",train=True,download=True,transform=ToTensor(),)
    test_data_mnist = datasets.MNIST(root="data",train=False,download=True,transform=ToTensor(),)
    train_mnist_dataloader = DataLoader(training_data_mnist, batch_size=batch_size)
    test_mnist_dataloader = DataLoader(test_data_mnist, batch_size=batch_size)
    fsp=Model()
    fsp.optimizer=[]
    fsp.loss_fn=[]
    fsp.device=[]
    loss=fsp.train(train_mnist_dataloader)
    fsp.test()


def train(data):
    train_features, train_labels = next(iter(data))
def learntrain(train_mnist_dataloader,test_mnist_dataloader,train_f_mnist_dataloader,test_f_mnist_dataloader):
    #Loading Models:
    ##################################################################################################################################
    arch = FSP_mnist(128).to(device)
    fsp= Model(arch,optimizer=[],loss=[],PATH="data/models/idm_FSP128_mnist.pt")

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
def run_multiple():
    for i in range(5):
        orig_stdout = sys.stdout
        subname=str(i)+"_"
        out="data/experiments/exp2/run1/"+subname+".txt"
        f = open(out, 'w')
        sys.stdout = f
        full_analysis_train(train_mnist_dataloader,test_mnist_dataloader,train_f_mnist_dataloader,test_f_mnist_dataloader)             
        sys.stdout = orig_stdout
        f.close()
