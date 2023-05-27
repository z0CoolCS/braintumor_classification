import torch.nn as nn
from torchvision import models


class ResNetPretrained(nn.Module):
    def __init__(self, 
                 num_classes,
                 tune_pretrained = False
                 ):
        super(ResNetPretrained, self).__init__()

        self.model = models.resnet18(weights="DEFAULT")

        if not tune_pretrained:
            print('Freezing parameters in pretrained model')
            for params in self.model.parameters():
                params.requires_grad = False

        self.model.fc = nn.Linear(512, num_classes)


    
    def forward(self, x):
        out = self.model(x)
        return out



