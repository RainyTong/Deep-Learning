from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
# from mlp_numpy import MLP
from pytorch_mlp import *
from sklearn import datasets
import matplotlib.pyplot as plt
import random
from torch.autograd import Variable

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 100
EVAL_FREQ_DEFAULT = 10

FLAGS = None

# Generate dataset
x,t = datasets.make_moons(1000)


# Show how x looks
plt.scatter(x[:,0],x[:,1])


# Devide training data and test data
train_x = x[:800]
train_t = t[:800]
test_x = x[800:]
test_t = t[800:]

# Show how test data x looks
plt.scatter(test_x[:,0],test_x[:,1])

def accuracy(predictions, targets):

    n_accu = 0
    for i in range(len(predictions)):
        if predictions[i][0][0] > predictions[i][0][1] and targets[i][0] == 0:
                n_accu = n_accu + 1
        elif predictions[i][0][0] < predictions[i][0][1] and targets[i][0] == 1:
                n_accu = n_accu + 1
    return (n_accu/len(predictions))

def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    n_inputs = len(train_x[0])
    
    n_hidden = list(map(int, DNN_HIDDEN_UNITS_DEFAULT.split()))

    n_classes = 2

    mlp = MLP(n_inputs, n_hidden, n_classes)
    
    losses = []
    predictions_train = []
    labels_train = []
    accu_train = []
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 

    optimizer = torch.optim.SGD(mlp.parameters(), lr=LEARNING_RATE_DEFAULT)
    
    for epoch in range(MAX_EPOCHS_DEFAULT):
        
        if epoch != 0 and epoch % (EVAL_FREQ_DEFAULT) == 0:
         
            accu = accuracy(predictions_train, labels_train)
            accu_train.append(accu)
            predictions_train = []
            labels_train = []
    
        
        for i in range(len(train_x)):
            input_x = torch.Tensor([train_x[i]])
            labels = torch.LongTensor([train_t[i]])
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = mlp(input_x)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()

    
        losses.append(loss.detach().numpy())
        predictions_train.append(outputs.detach().numpy())
        labels_train.append(labels.detach().numpy())
    
    return mlp, losses, accu_train

mlp, losses, accu_train = train()
plt.title('Loss Trend')

plt.plot(losses)
    
plt.xlabel('Epoch')
    
plt.ylabel('Loss')
    
plt.show()

plt.title('Train Accuracy')

plt.plot(accu_train)
    
plt.xlabel('Epoch')
    
plt.ylabel('Accuracy')
    
plt.show()


def test(mlp):
    predictions_test = []
    labels_test = []
    accu_test = []
    
    for i in range(len(test_x)):
        if i != 0 and i % EVAL_FREQ_DEFAULT == 0:
            accu = accuracy(predictions_test, labels_test)
            accu_test.append(accu)
            predictions_test = []
            labels_test = []
        
        input_x = torch.Tensor([test_x[i]])
        label_t = torch.Tensor([test_t[i]])
        
        out = mlp(input_x)
        
        predictions_test.append(out.detach().numpy())
        labels_test.append(label_t.detach().numpy())
    return accu_test

accu = test(mlp)
plt.title('Test Accuracy')

plt.plot(accu)
    
plt.xlabel('sample time')
    
plt.ylabel('accuracy')
    
plt.show()