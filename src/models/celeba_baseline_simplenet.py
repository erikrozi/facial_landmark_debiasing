import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append('../data')
sys.path.append('..')
sys.path.append('../utils')

from datasets import CelebaDataset
import landmark_transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer


class CNNSix(nn.Module):
    '''CNN-6 Implementation from https://arxiv.org/pdf/1711.06753.pdf'''
    def __init__(self, num_classes):
        """
        
        """
        super(CNNSix, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512*2*2, 1028)
        self.fc2 = nn.Linear(1028, num_classes)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))
        x = self.maxpool(F.relu(self.conv5(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    

data_loc = '/home/data/celeba/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = CelebaDataset(data_loc + 'landmarks_train.csv', data_loc + 'attr_train.csv', data_loc + 'images',
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]), 
                        landmark_transform=transforms.Compose([
                            landmark_transforms.Rescale(64),
                            #landmark_transforms.NormalizeLandmarks()
                        ]))
val_dataset = CelebaDataset(data_loc + 'landmarks_val.csv', data_loc + 'attr_val.csv', data_loc + 'images',
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]), 
                        landmark_transform=transforms.Compose([
                            landmark_transforms.Rescale(64),
                            #landmark_transforms.NormalizeLandmarks()
                        ]))
test_dataset = CelebaDataset(data_loc + 'landmarks_test.csv', data_loc + 'attr_test.csv', data_loc + 'images',
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]), 
                        landmark_transform=transforms.Compose([
                            landmark_transforms.Rescale(64),
                            #landmark_transforms.NormalizeLandmarks()
                        ]))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


num_classes = 10

model = CNNSix(num_classes)

trainer_params = {
    'model': model,
    'num_classes': 10,
    'train_loader': train_dataloader,
    'val_loader': val_dataloader,
    'test_loader': test_dataloader,
    'criterion': torch.nn.MSELoss,
    'criterion_args': {},
    'optimizer': torch.optim.Adam,
    'optimizer_args': {
         "lr": 1e-3,
         "weight_decay": 0
    },
    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler_args': {
        "factor": 0.3,
        "patience": 1,
        "cooldown": 1,
        "verbose": True
    },
    'scheduler_step_val': True,
    'debug': False,
}

trainer = Trainer(**trainer_params)

train_params = {
    'num_epochs': 20,
    'start_epochs': 0,
    'forward_args': {},
    'validate': True,
    'test': True,
    'save_dir': "../experiments/checkpoints",
    'tensorboard_dir': "../experiments/tensorboard",
    'log_dir': "../experiments/logs",
    'exp_name': 'celeba_baseline_simplenet_1',
    'epoch_per_save': 1,
    'epoch_per_print': 1,
    'batch_per_print': 100,
    'batch_per_save': 1000000,    # High number for no saving due to storage constraints
    'verbose': True
}

trainer.train(**train_params)

