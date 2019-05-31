from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        # Hidden Layers
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(n_inputs, n_hidden[0]))
        for k in range(len(n_hidden)-1):
            self.linears.append(nn.Linear(n_hidden[k], n_hidden[k+1]))
        
        # Output Layer
        self.out = nn.Linear(n_hidden[-1], n_classes)


    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x
        for linear_layer in self.linears:
            out = linear_layer(out)
            out = F.relu(out)

        out = self.out(out)
        return out
