import os

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

class Resnet18Network(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18Network, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        # return F.normalize(x, p=2, dim=1)
        return x
