"""
    Description: This library contains all the architectures used in the semantic utility paper
    Please refer to each of the functions/classes for a full description of what they do.

    Functions (True are the implemented Functions):
        

    Classes (True are the implemented classes):
        Atary_Enc
        CC_Enc
        Safety_Gym_Enc
        
            """

#Internal libraries
import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("[archs][rl]:su.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(full_path))))
#print("SU Internal libraries:")
from utils.matrices import *
from ml.functional.prob_fn import *
from ml.archs.sp import baselines as sp
#External libraries
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import *

class Ident(nn.Module):
    #Identity function those the same as lambda x: x
    def __init__(self,inout,parameters=None):
        inputs,outputs=inout
        super(Ident, self).__init__()
    def forward(self, x):
        return x
#Example:
#ident=Ident([20,18])
#x=torch.randn(4,20)
#print(x,ident(x))

class Rand(nn.Module):
    #Uniform random sampler from a uniform distribution
    def __init__(self,inout,parameters=None):
        inputs, outputs=inout
        super(Rand, self).__init__()
        self.outputs=outputs
    def forward(self, x):
        batch=x.size()[0]
        probs=torch.rand(batch,self.outputs)
        m=Categorical(probs)
        output = m.sample()
        return output
#Example:
#rand=Rand([20,18])
#x=torch.randn(4,20)
#print(rand(x))

class Simple_FSP_Enc(nn.Module):
    def __init__(self,inout,parameters):
        inputs, outputs=inout
        signals = parameters
        super(Simple_FSP_Enc, self).__init__()
        self.flatten = nn.Flatten()
        self.FSP =sp.FSP_pytorch(inout,parameters)
        self.layer_norm = nn.LayerNorm([outputs])

    def forward(self, x):
        x = self.flatten(x)
        x = self.FSP(x)
        x= self.layer_norm(x)
        return x
#Example:
#enc=Simple_FSP_Enc([32*32,4],64)
#x=torch.randn(4,1,32,32)
#print(enc(x))

class Simple_FSP_AC(nn.Module):
    def __init__(self,inout,parameters):
        input_act, output_act = inout["actor"]
        input_enc, output_enc = inout["enc"]
        self.classes = output_act
        super(Simple_FSP_AC, self).__init__()
        self.FSP_enc = Simple_FSP_Enc(inout["enc"],parameters["enc"])
        self.FSP_a = sp.FSP_pytorch(inout["actor"],parameters["actor"])
        self.FSP_c = sp.FSP_pytorch(inout["critic"],parameters["critic"])
        self.layer_norm = nn.LayerNorm([output_enc])
    def forward(self, x):
        x = self.FSP_enc(x)
        value = self.FSP_c(x)
        logits = self.FSP_a(x.detach())
        m=Categorical(logits=logits)
        sample , sample_grad =stg(m,logits,num_class=self.classes,dim=1,binary=False)
        return sample,sample_grad,value
#Example:
#inout={"enc":[64*64,8],"actor":[8,18],"critic":[8,1]}
#params={"enc":256,"actor":32,"critic":32}
#ac=Simple_FSP_AC(inout,params)
#x=torch.randn(4,1,64,64)
#action,action_grad,value =ac(x)
#print(action,value)
#print(action_grad.requires_grad)

class Squared_Image_Enc(nn.Module):
    def __init__(self,inout,parameters=None):
        inputs, outputs=inout
        channels, sqrd_dim_w, sqrd_dim_h = inputs
        if sqrd_dim_w==sqrd_dim_h:
            self.conv_layers=np.log(sqrd_dim_h)/np.log(2)
        else:
            raise ValueError("Input dims for Image show that image is not squared")
        super(Squared_Image_Enc, self).__init__()
        self.conv=[]
        self.conv.append(nn.Conv2d(channels, outputs, 4, stride=2,padding=1))
        for i in range(1,int(self.conv_layers)):
            self.conv.append(nn.Conv2d(outputs, outputs, 4, stride=2,padding=1))
        self.layer_norm = nn.LayerNorm([outputs,1,1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        for i in range(0,int(self.conv_layers)):
            x=self.conv[i](x)
        x = self.layer_norm(x)
        x = self.flatten(x)
        return x
#Example
#m = Squared_Image_Enc([[3,128,128],64])
#x = torch.randn(4,3,128,128)
#print(m(x))

class Att_Enc(nn.Module):
    def __init__(self,inout,parameters):
        inputs, outputs=inout
        channels, dim_w, dim_h = inputs
        self.att_layers = parameters
        self.att=[]
        for i in range(0,int(self.conv_layers)):
            self.att.append(nn.MultiheadAttention(embed_dim, num_heads))
        super(Att_Enc, self).__init__()
        self.conv=[]
        self.conv.append(nn.Conv2d(channels, outputs, 4, stride=2,padding=1))
        for i in range(1,int(self.conv_layers)):
            self.conv.append(nn.Conv2d(outputs, outputs, 4, stride=2,padding=1))
        self.layer_norm = nn.LayerNorm([outputs,1,1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        for i in range(0,int(self.att_layers)):
            x=self.conv[i](x)
        x = self.layer_norm(x)
        x = self.flatten(x)
        return x
#Example
#multihead_attn = nn.MultiheadAttention(embed_dim=3,num_heads=1,kdim=3,vdim=3,batch_first=True)
#target=torch.ones(4,1,1)
#target=torch.tensor([[[1.]],[[0.]],[[0.]],[[1.]]])
#x=torch.randn(2,3,8,8)
#print(x.size())
#flat=nn.Flatten(start_dim=2)
#x=flat(x)
#x=torch.permute(x, (0, 2, 1))
#print(x.size())
#s=torch.rand()
#mask=torch.tensor([[True,True,False,False]])
#attn_output, attn_output_weights = multihead_attn(x,x,x)
#print(target)
#print(attn_output)
#print(attn_output.size(),attn_output_weights.size())

class Att_FSP_Enc(nn.Module):
    def __init__(self,inout,parameters):
        inputs, outputs=inout
        channels, dim_w, dim_h = inputs
        self.att_layers = parameters
        super(Att_FSP_Enc, self).__init__()
        self.conv=[]
        self.conv.append(nn.Conv2d(channels, outputs, 4, stride=2,padding=1))
        for i in range(1,int(self.conv_layers)):
            self.conv.append(nn.Conv2d(outputs, outputs, 4, stride=2,padding=1))
        self.layer_norm = nn.LayerNorm([outputs,1,1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        for i in range(0,int(self.conv_layers)):
            x=self.conv[i](x)
        x = self.layer_norm(x)
        x = self.flatten(x)
        return x

class RNN(nn.Module):
    def __init__(self,inout,parameters=[]):
        inputs,outputs=inout
        super(RNN, self).__init__()
        self.rnn = nn.GRUCell(inputs, outputs)
        self.hn=torch.zeros(outputs)

    def forward(self, x):
        output=self.rnn(x,self.hn)
        self.hn=output
        return output

#test=RNN([3,2])
#input=torch.randn(5,3)
#h0=torch.randn(5,2)
#test.hn=h0
#for i in range(0,10):
#    print(test(input))

"""Probabilistic Models"""

class ProbModel(nn.Module):
    """Probabilistic model used for defining Discrete probabilistic Models :
    Supported: MultivariateBernoulli ,UnivariateCategorical
    """
    def __init__(self,inout,parameters):
        inputs,outputs=inout
        signals,binary=parameters
        super(ProbModel, self).__init__()
        self.fsp = sp.FSP_pytorch([inputs,outputs],signals)
        self.binary=binary
        self.outputs=outputs
    def forward(self, x):
        logits = self.fsp(x)
        if self.binary:
            m=Bernoulli(logits=logits)
            sample=stg(m,logits,binary=self.binary)
        else:
            m=Categorical(logits=logits)
            sample , sample_grad =stg(m,logits,num_class=self.outputs,dim=1,binary=self.binary)
        #l_prob=m.log_prob(logits)
        return sample ,sample_grad

#EXAMPLE
#test=ProbModel([3,4],[128,False])
#input=torch.randn(3,3)
#sample ,sample_grad=test(input)
#print(sample,sample_grad.size())

class DensFuncModel(nn.Module):
    """Probabilistic model used for defining Discrete probabilistic Models :
    Supported: UnivariateNormal,MultivariateDiagonalNormal
    """
    def __init__(self,inout,parameters):
        inputs,outputs=inout
        signals,binary=parameters
        super(DensFuncModel, self).__init__()
        self.fsp = sp.FSP_pytorch(signals,inputs,outputs)
        self.std=torch.ones(outputs)
        self.outputs=outputs
    def forward(self, x):
        logits = self.FSP(x)
        if binary:
            m=Normal(logits,self.std)
            sample=m.rsample()
        else:
            m = MultivariateNormal(logits, scale_tril=torch.diag(self.std))
            sample=m.rsample()
        l_prob=m.log_prob(logits)
        return sample,l_prob
#EXAMPLE
#test=DenseFuncModel([3,4],[128,False])
#input=torch.randn(3,3)
#sample ,sample_grad=test(input)
#print(sample,sample_grad.size())


class MultivarProbModel(nn.Module):
    def __init__(self,inout,parameters):
        inputs,outputs=inout
        vatriables,signals=parameters
        super(MultivarProbModel, self).__init__()
        self.var=variables
        self.mfsp=[variables]
        for n in range(0,variables):
            self.mfsp[n] = sp.FSP_pytorch(signals,inputs,outputs)
        
    def forward(self, x):
        sing_logits=[self.var]
        mult_logits=torch.Tensor([])
        for n in range(0,self.var):
            sing_logits[n] = self.mfsp[n](x)
            torch.cat((multi_logits, sing_logits[n]),0)
        return logits  

class MultivarProbModelv2(nn.Module):
    def __init__(self,inout,parameters):
        inputs,outputs=inout
        vatriables,signals=parameters
        super(MultivarProbModelv2, self).__init__()
        self.var=variables
        self.mfsp=[variables]
        for n in range(0,variables):
            self.mfsp[n] = sp.FSP_pytorch(signals,inputs,outputs)
        
    def forward(self, x):
        sing_logits=[self.var]
        mult_logits=torch.Tensor([])
        for n in range(0,self.var):
            sing_logits[n] = self.mfsp[n](x)
            torch.cat((multi_logits, sing_logits[n]),0)
        return logits  

class SimpleEdge(nn.Module):
    def __init__(self,inout,parameters):
        self.inputs,outputs=inout
        self.idx , self.detach =parameters
        super(SimpleEdge, self).__init__()

    def forward(self, x ):
        y=torch.tensor([])
        if len(self.idx)>1:#Loop when idx is bigger than one (Concatenate)
            for idx,n_tensor in enumerate(x):
                if idx in self.idx:
                    if len(n_tensor.size())>2:
                        raise ValueError("Concatenate only works with Tensors of dimention 1")
                    else:
                        y=torch.cat((y,n_tensor),-1)
            logits=y
        else:#Return the idx value
            logits=x[self.idx]
        if self.detach:
            return logits.detach()
        else:
            return logits
#Example
#arch=SimpleEdge([4,1],[torch.tensor([0,2]),False])
#x=torch.randn(2,2)
#x1=torch.randn(2,3)
#x2=torch.randn(2,1)
#z=[x,x1,x2]
#print(z)
#print(arch(z))
