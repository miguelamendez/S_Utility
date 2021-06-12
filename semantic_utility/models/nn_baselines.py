"""[Neural Network Architectures for all Agents parts]"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import numpy as np
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNNStateEncoderBaselineNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        """[Neural Network for processng images]
        Args:
            input_dims ([tuple]): [dimention of the input size of image] . 
            output_dims ([type]): [dimention of the output encoded state] 
        """  
        super(CNNStateEncoderBaselineNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(
                7 * 7 * 64,
                output_dims),
            nn.LeakyReLU()
        )
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, x):
        """[forward function]
        Args:
            x ([tensor batch]): [The state of the environment assumed to be an Image] .
        """  
        x = self.model(x)
        return x

class LinearStateEncoderBaselineNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        """[Neural Network for processing 1 dimentional data types]

        Args:
            input_dims ([tuple]): [dimention of the input ] . 
            output_dims ([type]): [dimention of the output encoded state].
        """  
        super(LinearStateEncoderBaselineNetwork, self).__init__()
        self.model =  nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256,output_dims),
            nn.LeakyReLU()
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, x):
        """[forward function]
        Args:
            x ([type]): [State of the environment assumed to ve a 1d vector] .
        """  
        x = self.model(x)
        return x

class CNNStateDecoderBaselineNetwork(nn.Module):
    def __init__(self,input_dims,output_dims ):
        super(CNNStateDecoderBaselineNetwork, self).__init__()
        """[not implemented]"""
    def forward(self, x):
        return x

class LinearStateDecoderBaselineNetwork(nn.Module):
    def __init__(self,input_dims,output_dims ):
        super(LinearStateDecoderBaselineNetwork, self).__init__()
        """[not implemented]"""
    def forward(self, x):
        return x

class LinearActorBaselineNetwork(nn.Module):
    def __init__(self,input_dims,output_dims):
        """[Neural Network for choosing actions assumed to be descrete actions]
        Args:
            input_dims ([tuple]): [dimention of the input (encoded state)] . 
            output_dims ([int]) : [dimention of the output (probability distribution over the actions)]
        """  
        super(LinearActorBaselineNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, 512),
            nn.LeakyReLU(),
            nn.Linear(512, output_dims),
            nn.Softmax(dim=-1)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, x):
        """[forward function]
        Args:
            x ([type]): [Tipically the state or encoded state depending on the agent] .
        """  
        x = self.model(x)
        return x

class LinearValueBaselineNetwork(nn.Module):
    def __init__(self, input_dims):
        """[Neural Network for predicting the Value functions]

        Args:
            input_dims ([tuple]): [dimention of the input (encoded state)] .
        """  
        super(LinearValueBaselineNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, x):
        """[forward function]
        Args:
            x ([type]): [Tipically the state or encoded state depending on the agent] .
        """  
        x = self.model(x)
        return x

class LinearActionValueBaselineNetwork(nn.Module):

    def __init__(self, input_dims):
        """[Neural Network for calculating the Action-Value function]

        Args:
            input_dims ([tuple]): [dimention of the input (encoded state,action)] . 
        """  
        super(LinearActionValueBaselineNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, inputs):
        """[forward function]
        Args:
            inputs ([type]): [Encoded state and action performed by the agent] .
        """  
        enc_state , action = inputs
        x = torch.cat((enc_state, action), 1)
        x = self.model(x)
        return x

class LinearInverseBaselineNetwork(nn.Module):

    def __init__(self, input_dims,output_dims):
        """[Neural Network for defining an Inverse Model]
        Args:
            input_dims ([tuple]): [dimention of the input] . The input size is equal to the encoded_state x2
            arg ([type]): [dimention of the output] .  The output size is equal to the number of possible actions
        """ 
        super(LinearInverseBaselineNetwork, self).__init__()
        self.model =  nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256,output_dims),
            nn.Softmax(dim=-1)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, inputs):
        """[forward function]
        Args:
            inputs:
            enc_state ([tensor]): [State given by the environment] .
            enc_n_state ([tensor]): [Next State given by the environment] .
        """  
        enc_state, enc_next_state = inputs
        x = torch.cat((enc_state, enc_n_state), 1)
        x = self.model(x)
        return x

class ForwardICMBaselineNetwork(nn.Module):
    def __init__(self, input_dims,use_cuda=True):
        super(ForwardICMBaselineNetwork, self).__init__()
        """[Simple Forward Model used in the Curiosity paper , it has a resnet and 2 forward nets. Modified from source: https://github.com/jcwleo/curiosity-driven-exploration-pytorch]
        Args:
            action_dims ([tuple]): [enc_state dims]
            enc_state_dims ([int]): [action_dims]
            
        """
        enc_state_dims,action_dims=input_dims
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.residual = [nn.Sequential(
            nn.Linear(enc_state_dims + action_dims, 512),
            nn.LeakyReLU(),
            nn.Linear(512, enc_state_dims),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(enc_state_dims + action_dims, enc_state_dims),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(enc_state_dims + action_dims, enc_state_dims),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        """[summary]
        Args:
            inputs ([tuple]): [ inputs : encoded state and action ] .
        """
        enc_state , action = inputs
        pred_next_state_feature_orig = torch.cat((enc_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)
        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig
        x = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))
        return x

class ForwardDWMBaselineNetwork(nn.Module):
    def __init__(self,input_dims ):
        super(ForwardDWMBaselineNetwork, self).__init__()
        """[not implemented]"""
    def forward(self, inputs):
        return outputs

class ImageBaselineNetwork(nn.Module):
    def __init__(self,input_dims,output_dims ):
        super(ImageBaselineNetwork, self).__init__()
        """[not implemented]"""
    def forward(self, inputs):
        return outputs

class DeepWeightNetwork(nn.Module):
    def __init__(self, input_dims,output_dims):
        """[Neural Network for defining an Inverse Model]
        Args:
            input_dims ([tuple]): [dimention of the input] . The input size is equal to the encoded_state+action
            outpuit_dims ([type]): [dimention of the output deep weights] .  The output size is equal to the number of literals
        """ 
        enc_state_dims,action_dims=input_dims
        super(DeepWeightNetwork, self).__init__()
        self.model =  nn.Sequential(
            nn.Linear(enc_state_dims+1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,output_dims),
            nn.Softmax(dim=-1)
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, inputs):
        """[forward function]
        Args:
            inputs:
            enc_state ([tensor]): [State given by the environment] .
            enc_n_state ([tensor]): [Next State given by the environment] .
        """  
        enc_state, action = inputs
        outputs = torch.cat((enc_state, action), 1)
        outputs = self.model(outputs)
        return outputs
