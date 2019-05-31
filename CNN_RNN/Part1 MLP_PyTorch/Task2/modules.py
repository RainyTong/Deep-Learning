import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.params = {}
        self.grads = {}
        self.sum_grads = {}
        self.params['weight'] = np.random.normal(loc=0, scale=0.0001, size=(in_features, out_features))
        self.params['bias'] = 0
        self.grads['weight'] = np.zeros((in_features, out_features))
        self.grads['bias'] = 0
        self.sum_grads['weight'] = np.zeros((in_features, out_features))
        self.sum_grads['bias'] = 0
        self.x = 0
        
        
    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """

        self.x = x
       
        out = np.dot(self.x, self.params['weight']) + self.params['bias']
       
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """

        db = dout
        dw = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.params['weight'].T)  

        self.grads['weight'] = dw
        self.grads['bias'] = db
        self.sum_grads['weight'] += self.grads['weight']
        self.sum_grads['bias'] += self.grads['bias']
        return dx

class ReLU(object):
    def __init__(self):
        self.x = 0
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.x = x
        z = np.zeros(len(x))
        out = np.maximum(x, z)
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """

        g = np.zeros((1, len(self.x[0])))
        for i in range(len(self.x[0])):
            if self.x[0][i] > 0:
                g[0][i] = 1
            else:
                g[0][i] = 0
       
        dx = g * dout
        return dx

class SoftMax(object):
    
    def exp_normalize(self, x):
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()  
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        out = self.exp_normalize(x)
        return out

    def backward(self, out, label_t):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
     
        dx = out - label_t
        
        return dx

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        out = np.sum(- y * np.log(x))
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        dx = -y/x
        return dx
