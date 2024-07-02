import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from utils import save_config_file, accuracy, save_checkpoint
import torch
import torch.nn as nn
import torchvision.models as models


torch.manual_seed(850011)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        # self.args = kwargs['args']
        backbone = models.resnet18(pretrained=False)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        # self.optimizer = kwargs['optimizer']
        # self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter()
        
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def forward(self,x):
        return self.model(x)

class SimCLR_cls(nn.Module):
    def __init__(self, num_classes):
        super(SimCLR_cls, self).__init__()
        self.backbone = SimCLR()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x.view(x.size(0), -1))
# x = torch.randn(16, 3, 224, 224)
# model = SimCLR()
# y = model(x)
# print(y.shape)