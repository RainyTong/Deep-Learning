from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class NP_MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.n_inputs = n_inputs # 2
        self.n_hidden = n_hidden # [20]
        self.n_classes = n_classes # 2
        
        self.n_h_layers = len(n_hidden)
        self.linear_layers = []
        self.relu_layers = []
        
        in_size = n_inputs
        for i in range(self.n_h_layers):
            out_size = n_hidden[i]
            linear = Linear(in_size, out_size)
            self.linear_layers.append(linear)
            in_size = out_size
            
            relu = ReLU()
            self.relu_layers.append(relu)
        
        out_size = n_classes
        linear = Linear(in_size, out_size)
        self.linear_layers.append(linear)
        
    
    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network  # ==> [ 0.10942261,  0.04516827]
        Returns:
            out: output of the network # ==> [0.9 0.1]
        """
       
        input_x = x
        
        # hidden layers forwarding
        for i in range(self.n_h_layers):
            
            linear = self.linear_layers[i]
            
            linear_out = linear.forward(input_x)
            
            relu = self.relu_layers[i]
            
            relu_out = relu.forward(linear_out)

            input_x = relu_out
        
        # output layer forwarding
        linear = self.linear_layers[self.n_h_layers]
      
        linear_out = linear.forward(input_x)

        softmax = SoftMax()
        
        out = softmax.forward(linear_out)

        return out

    def backward(self, out, label_t):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """

        softmax = SoftMax()
        dx = softmax.backward(out, label_t)
        
        for i in range(len(self.linear_layers)-1, 0, -1):
            linear = self.linear_layers[i]

            dx = linear.backward(dx)

            relu = self.relu_layers[i-1]

            dx = relu.backward(dx)

        linear = self.linear_layers[0]
        linear.backward(dx)
        
        return