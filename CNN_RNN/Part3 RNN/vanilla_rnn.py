from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        # for hidden layer
        self.W_hx = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hh = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_h = Parameter(torch.Tensor(hidden_dim))
        
        # for output layer
        self.W_ph = Parameter(torch.Tensor(hidden_dim, output_dim))
        self.b_o = Parameter(torch.Tensor(output_dim))
       
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x):
        # Implementation here ...
        
        h_t = torch.zeros(self.hidden_dim)
        output = 0
            
        for t in range(self.seq_length): # iterate over the time steps
            x_t = x[:, t].view(128,-1)
           
            h_t = F.tanh(x_t @ self.W_hx + h_t @ self.W_hh + self.b_h)
            
        output = h_t @ self.W_ph + self.b_o
        y = F.softmax(output)
        
        return y
        
    # add more methods here if needed
