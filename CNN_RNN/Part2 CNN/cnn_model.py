from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
    
    self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
    self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
    
    self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
    self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
    
    self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
    self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)
    
    self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    self.pool5= nn.MaxPool2d(3, stride=2, padding=1)
    
    self.linear = nn.Linear(512, 10)
    
  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    print("-------")
    x = self.pool1(F.relu(self.conv1(x)))
    
    x = self.pool2(F.relu(self.conv2(x)))
    
    x = self.pool3(F.relu(self.conv4(F.relu(self.conv3(x)))))
    
    x = self.pool4(F.relu(self.conv6(F.relu(self.conv5(x)))))
    
    x = self.pool5(F.relu(self.conv8(F.relu(self.conv7(x)))))
    print(x.shape)
    x = x.view(-1, 512)
    print(x.shape)
    out = self.linear(x)
    
    return out
