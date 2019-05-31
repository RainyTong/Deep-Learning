from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.argv=['']
del sys

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from dataset import PalindromeDataset
from lstm import LSTM

def accuracy(outputs, batch_targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """

    _, predicted = torch.max(outputs.data, 1)
    total = batch_targets.size(0)
    correct = (predicted == batch_targets).sum().item()
    accuracy = 100.0 * correct / total
    
    return accuracy

def train(config, input_length):

    # Initialize the model that we are going to use
    model = LSTM(input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size)  # fixme

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # fixme
#     optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    losses = []
    accuracies = []
    loss = 0.0


    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Add more code here ...
        optimizer.zero_grad() 
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)   

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

        # Add more code here ...

        loss += loss.item()   # fixme
        accu = 0.0  # fixme
        

        if step % 10 == 0:
            # print acuracy/loss here
            print('[step: %5d] loss: %.4f' %
                          (step, loss / 10))
            losses.append(loss / 10)
            loss = 0.0
            accu = accuracy(outputs, batch_targets)
            accuracies.append(accu)
            print('Accuracy on training dataset: %.3f %%' % (accu))


        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    
    return model, losses, accuracies

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    

model, losses, accuracies = train(config, config.input_length)

plt.title('Loss Trend')

plt.plot(losses)
    
plt.xlabel('Sample time')
    
plt.ylabel('Loss')
    
plt.show()


plt.title('Train Accuracy')

plt.plot(accuracies)
    
plt.xlabel('Sample time')
    
plt.ylabel('Accuracy (%)')
    
plt.show()

def test(model, config, input_length):
    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    accuracies = []
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        outputs = model(batch_inputs)
        accu = 0.0  
        
        if step % 10 == 0:
            accu = accuracy(outputs, batch_targets)
            accuracies.append(accu)
            
        if step == 2000:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done testing.')
    
    return accuracies

test_accuracies = test(model, config, config.input_length)

plt.title('Test Accuracy')

plt.plot(test_accuracies)
    
plt.xlabel('Sample time')
    
plt.ylabel('Accuracy (%)')
    
plt.show()

print("Palindromes Length: T = 11")
print("Average accuracy over 2000 sampled test: " + str(np.mean(test_accuracies)) + " %")