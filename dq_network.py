import torch
from torch  import nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """
        Instantiate Neural Network to approximate action value function

        Arguments:

            state_size (int): Demension of environment state vector
            action_size (int): Demension of agent action vector
            seed (int): Random seed for reproducability 
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        """
        Deep network with batch normalization 
        between fully connected layers
        """
        
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, action_size)

    def forward(self, state):
        """
        Perform a forward propagation inference on environment state vector

        Arguments:
            state - the enviornment state vector

        Returns - action value
        """

        state = self.fc1(state)
        state = self.bn1(state)
        state = F.leaky_relu(state)
        
        state = self.fc2(state)
        state = self.bn2(state)
        state = F.leaky_relu(state)
        
        state = self.fc3(state)
        state = self.bn3(state)
        state = F.leaky_relu(state)
        
        state = self.fc4(state)
        state = self.bn4(state)
        state = F.leaky_relu(state)
        
        state = self.fc5(state)       

        return state