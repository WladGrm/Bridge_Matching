import torch
import torch.nn as nn
from sklearn.datasets import make_swiss_roll
import torch.nn.init as init

from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt



#Normal distribution Sampler
def sample_2d_gaussian(batch_size=32, device = "cpu"):
    """
    Samples a batch of (x,y) points from Gaussian(Normal) distribution
    """
    return torch.randn(batch_size, 2, device = device)


#Swiss Roll distribution Sampler
def sample_2d_swiss_roll(batch_size=32, std=0.1, scaler=0.1, device = "cpu"):
    """
    Samples a batch of (x,y) points from Swiss Roll distribution
    """
    return (scaler*torch.tensor(make_swiss_roll(n_samples=batch_size, noise=std)[0][:, [0, 2]]).float()).to(device)



def train_velocity_forward(vnet, eps=0.1, epoches = 1000, batch_size = 64, l_r = 1e-3, std=0.5, device = "cpu"):
    """
    Velocity/Drift network training from Gaussian (Normal) to Swiss Roll distribution
    """

    #vnet.reset_parameters()
    vnet = vnet.to(device)
    
    #Optimizer 
    vnet_opt = torch.optim.Adam(vnet.parameters(), lr=l_r)

    loss_arr = []
    for i in tqdm(range(epoches)):


        t = torch.rand(size=(batch_size, 1)).to(device) #Sample t ~ Uniform(0,1)

        x_0 = sample_2d_gaussian(batch_size=batch_size).to(device) #Sample a batch of gaussians N ~ (0,I)

        x_1 = sample_2d_swiss_roll(batch_size=batch_size, std=std).to(device) #Sample a batch of swiss roll points

        #Calculate x_t as linear interpolation with added noise N ~ (0, eps*t*(1-t)*I)
        added_noise = torch.sqrt(eps * t * (1. - t)) * torch.randn_like(x_0)
        x_t = t * x_1 + (1. - t) * x_0 + added_noise

        #=======
        #DATA PREP 
        #x_t [BS,[x,y]] --> x_tt [BS,[x,y,t]]
        x_tt = torch.cat((x_t, t), dim = 1).to(device)
        v_target = ((x_1 - x_t) / (1. - t)).to(device)
        #v_target = x_1-x_0
        predicted_v = vnet(x_tt)
        #Loss 
        loss_function = nn.MSELoss()
        loss = loss_function(predicted_v, v_target)

        with torch.no_grad():
            loss_arr.append(loss.cpu().detach().numpy())
        
        #Training network
        #vnet.train(True)
        vnet_opt.zero_grad(); loss.backward(); vnet_opt.step()


        #if i % 10 == 0:
            #clear_output(wait=True)
            #print("Step", i)
            #print("Loss", loss_arr)


    return loss_arr






def train_velocity_backward(vnet, eps=0.1, epoches = 1000, batch_size = 64, l_r = 1e-3, std=0.5, device = "cpu"):
    """
    Velocity/Drift network training from Swiss Roll distribution to Gaussian (Normal) distribution
    """

    #vnet.reset_parameters()
    vnet = vnet.to(device)
    
    #Optimizer 
    vnet_opt = torch.optim.Adam(vnet.parameters(), lr=l_r)

    loss_arr = []
    for i in tqdm(range(epoches)):


        t = torch.rand(size=(batch_size, 1)).to(device) #Sample t ~ Uniform(0,1)

        x_1 = sample_2d_gaussian(batch_size=batch_size).to(device) #Sample a batch of gaussians N ~ (0,I)

        x_0 = sample_2d_swiss_roll(batch_size=batch_size, std=std).to(device) #Sample a batch of swiss roll points

        #Calculate x_t as linear interpolation with added noise N ~ (0, eps*t*(1-t)*I)
        added_noise = torch.sqrt(eps * t * (1. - t)) * torch.randn_like(x_0)
        x_t = t * x_1 + (1. - t) * x_0 + added_noise

        #=======
        #DATA PREP 
        #x_t [BS,[x,y]] --> x_tt [BS,[x,y,t]]
        x_tt = torch.cat((x_t, t), dim = 1).to(device)
        v_target = ((x_1 - x_t) / (1. - t)).to(device)
        #v_target = x_1-x_0
        predicted_v = vnet(x_tt)
        #Loss 
        loss_function = nn.MSELoss()
        loss = loss_function(predicted_v, v_target)

        with torch.no_grad():
            loss_arr.append(loss.cpu().detach().numpy())
        
        #Training network
        #vnet.train(True)
        vnet_opt.zero_grad(); loss.backward(); vnet_opt.step()


        #if i % 10 == 0:
            #clear_output(wait=True)
            #print("Step", i)
            #print("Loss", loss_arr)


    return loss_arr






#Multilayer perceptron class with ReLu activations
class MLP(nn.Module):
    def __init__(self, layer_list, use_batch_norm=False, dropout_prob=0.0):
        """
        Initialize the Multilayer Perceptron model.

        Args:
            layer_list (list): a list to configure layers of the neural network.
            use_batch_norm (bool): If True, applies batch normalization after each hidden layer (except the last layer).
            dropout_prob (float): The probability of dropping out neurons during training.
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        self.dropout_prob = dropout_prob
        
        for i in range(len(layer_list) - 1):
            self.layers.append(nn.Linear(layer_list[i], layer_list[i + 1]))
            if use_batch_norm and i < len(layer_list) - 2:
                self.layers.append(nn.BatchNorm1d(layer_list[i + 1]))
            if dropout_prob > 0 and i < len(layer_list) - 2:
                self.layers.append(nn.Dropout(p=dropout_prob))

        print('Vnet params:', np.sum([np.prod(p.shape) for p in self.layers.parameters()]))
    
    def forward(self, x):
        """
        Define the forward pass of the network.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output of the network after passing through all layers.
        """
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)  # to remove ReLu from last layer
        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                layer.reset_parameters()
            elif isinstance(layer, nn.Dropout):
                pass







class SDE_Solver(nn.Module):
    def __init__(self, vnet, steps=100, start_t = 0, end_t = 1, eps=0.1, device="cpu"):
        """
        Initializing Euler-Maruyama Solver for SDE
        """
        super().__init__()
        self.device = device
        self.vnet = vnet #trained neural network for velocity/drift estimation
        self.eps = torch.tensor(eps).to(self.device) #if eps = 0 then ODE
        self.start_t = start_t
        self.end_t = end_t
        self.steps = steps #number of steps for time discretization
        self.delta_t = float((self.end_t-self.start_t)) / steps #since time has values in range [start_t, end_t]
        


    def get_velocity(self, x, t):
        self.vnet.eval()
        x_tt = torch.cat((x, t), dim = 1).to(self.device)
        return self.vnet(x_tt).to(self.device)
    

    def get_velocity_2(self, x, t, vnet_backward):
        self.vnet.eval()
        vnet_backward.eval()
        x_tt = torch.cat((x, t), dim = 1).to(self.device)
        return (self.vnet(x_tt).to(self.device) - vnet_backward(x_tt)) / 2
    
    def get_noise(self, x):
        return torch.randn_like(x).to(self.device) #N ~ (0, I)
        

    def solve(self, x, printable=True):
        """
        Euler Maruyama method

        Args:
            x (Tensor [BS, N=2]): -- initial distribution to be mapped.

        Returns:
            trajectory_torched (Tensor [BS, steps, N=2]): trajectory of points.
            prediction (Tensor [BS, N=2]): mapped distribution.
        """
        x = x.to(self.device) #Torch Tensor shape [BS, 2]
        batch_size = x.shape[0]
        t = torch.zeros((batch_size,1), device=self.device) #Torch Tensor filled with 0s shape [BS, 1]
        

        trajectory = [x]

        for i in range (self.steps):
            with torch.no_grad():
                noise = self.get_noise(x)
                velocity = self.get_velocity(x, t)
            
                x = x + velocity * self.delta_t + torch.sqrt(self.eps * self.delta_t).to(self.device) * noise
                t+= self.delta_t

                trajectory.append(x)
        
        trajectory_torched = torch.stack(trajectory, dim=1)
        prediction = trajectory_torched[:,-1,:]

        if printable==True:
            return trajectory_torched.detach().cpu(), prediction.detach().cpu()
        else:
            return trajectory_torched, prediction
        

    def solve_2(self, x, vnet_backward, printable=True):
        """
        Euler Maruyama method

        Args:
            x (Tensor [BS, N=2]): -- initial distribution to be mapped.

        Returns:
            trajectory_torched (Tensor [BS, steps, N=2]): trajectory of points.
            prediction (Tensor [BS, N=2]): mapped distribution.
        """
        vnet_backward = vnet_backward.to(self.device)
        x = x.to(self.device) #Torch Tensor shape [BS, 2]
        batch_size = x.shape[0]
        t = torch.zeros((batch_size,1), device=self.device) #Torch Tensor filled with 0s shape [BS, 1]
        

        trajectory = [x]

        for i in range (self.steps):
            with torch.no_grad():
                noise = self.get_noise(x)
                velocity = self.get_velocity_2(x, t, vnet_backward)
            
                x = x + velocity * self.delta_t + torch.sqrt(self.eps * self.delta_t).to(self.device) * noise
                t+= self.delta_t

                trajectory.append(x)
        
        trajectory_torched = torch.stack(trajectory, dim=1)
        prediction = trajectory_torched[:,-1,:]

        if printable==True:
            return trajectory_torched.detach().cpu(), prediction.detach().cpu()
        



















