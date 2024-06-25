
import torch
import os
import torch.nn as nn
from torchvision import models

def create_model(mo, n_classes):

    model = resnet(mo.lower(), n_classes)

    return model

def resnet(mo, nc, pretrain=True):
    
    if mo == 'resnet18':
        model = models.resnet18(pretrained=pretrain)
    elif mo == 'resnet32':
        model = models.resnet32(pretrained=pretrain)
    elif mo == 'resnet50':
        model = models.resnet50(pretrained=pretrain)   
    elif mo == 'alexnet':
        model = models.alexnet(pretrained=pretrain)
        model.classifier[6] = torch.nn.Linear(4096, nc)
    
    if mo in ['resnet18','resnet32', 'resnet50']:
        model.fc = torch.nn.Linear(model.fc.in_features, nc)
        
    return model

# model = create_model('resnet18', n_classes=10)
# print(model)

# for name, param in model.named_parameters():
#     print(name)