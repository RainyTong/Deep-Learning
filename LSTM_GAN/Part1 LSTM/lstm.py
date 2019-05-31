from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
             
        # ---for hidden layer---
        
        # for input modulation gate g:
        self.W_gx = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_gh = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = Parameter(torch.Tensor(hidden_dim))
        
        # for input gate i:
        self.W_ix = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_ih = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = Parameter(torch.Tensor(hidden_dim))
        
        # for forget gate f:
        self.W_fx = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_fh = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = Parameter(torch.Tensor(hidden_dim))
        
        # for output gate o:
        self.W_ox = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_oh = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = Parameter(torch.Tensor(hidden_dim))
        
        
        # ---for output layer---
        self.W_ph = Parameter(torch.Tensor(hidden_dim, output_dim))
        self.b_p = Parameter(torch.Tensor(output_dim))
       
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.orthogonal_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x):
        # Implementation here ...
        
        h_t = torch.zeros(self.hidden_dim)
        c_t = torch.zeros(self.hidden_dim)
        output = 0
            
        for t in range(self.seq_length): # iterate over the time steps
            x_t = x[:, t].view(128,-1)
            g_t = F.tanh(x_t @ self.W_gx + h_t @ self.W_gh + self.b_g)
            i_t = F.sigmoid(x_t @ self.W_ix + h_t @ self.W_ih + self.b_i)
            f_t = F.sigmoid(x_t @ self.W_fx + h_t @ self.W_fh + self.b_f)
            o_t = F.sigmoid(x_t @ self.W_ox + h_t @ self.W_oh + self.b_o)
            c_t = g_t * i_t + c_t * f_t
            h_t = F.tanh(c_t) * o_t
                       
        output = h_t @ self.W_ph + self.b_p
        y = F.softmax(output)
        
        return y
        
    # add more methods here if needed
