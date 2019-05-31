from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 100
MAX_EPOCHS_DEFAULT = 4
EVAL_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])
trainset = dset.CIFAR10('./data', train=True, transform=transform, target_transform=None, download=False)
testset = dset.CIFAR10('./data', train=False, transform=transform, target_transform=None, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, num_workers=2)

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
# show some random training images
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
dataiter = iter(trainloader)
images, labels = dataiter.next()
 
# print images
imshow(torchvision.utils.make_grid(images))
# print labels
print('     '.join('%5s'%classes[labels[j]] for j in range(BATCH_SIZE_DEFAULT)))
print()

print("trainloader lengh: "+str(len(trainloader)))
print("dataiter lengh: "+str(len(dataiter)))

def accuracy(outputs, batch_y):
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
    total = batch_y.size(0)
    correct = (predicted == batch_y).sum().item()
    accuracy = 100.0 * correct / total
    
    
    return accuracy

def train():
    """
    Performs training and evaluation of CNN model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    n_channels = 3
    n_classes = 10
    
    cnn = CNN(n_channels, n_classes)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cnn.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE_DEFAULT)
    
    losses = []
    accuracies = []
    
    for epoch in range(MAX_EPOCHS_DEFAULT):
        timestart = time.time()
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(trainloader):
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward + Backward + Optimize
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = cnn(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % EVAL_FREQ_DEFAULT == EVAL_FREQ_DEFAULT-1:  
                print('[epoch: %d, step: %5d] loss: %.4f' %
                          (epoch, step, running_loss / EVAL_FREQ_DEFAULT))
                losses.append(running_loss / EVAL_FREQ_DEFAULT)
                running_loss = 0.0
                accu = accuracy(outputs, batch_y)
                accuracies.append(accu)
                print('Accuracy on the %d train images: %.3f %%' % (batch_y.size(0),
                            accu))
  
        
        print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))
    
    print('---Finished Training---')
    
    return cnn, losses, accuracies

cnn, losses, accuracies = train()

def test(cnn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cnn.to(device)
    
    test_accuracies = []
    for step, (batch_x, batch_y) in enumerate(testloader):
        
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
        outputs = cnn(batch_x)
        
        if step % 10 == 9:
            accu = accuracy(outputs, batch_y)
            test_accuracies.append(accu)
            print('Accuracy on the %d test images: %.3f %%' % (batch_y.size(0),
                            accu))

    print('---Finished Testing---')   
    return test_accuracies

test_accuracies = test(cnn)

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


plt.title('Test Accuracy')

plt.plot(test_accuracies)
    
plt.xlabel('Sample time')
    
plt.ylabel('Accuracy (%)')
    
plt.show()