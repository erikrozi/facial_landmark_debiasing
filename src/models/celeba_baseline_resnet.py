import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch import nn

import sys
sys.path.append('../data')
sys.path.append('..')
sys.path.append('../utils')

from datasets import CelebaDataset
import landmark_transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer

data_loc = '/home/data/celeba/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = CelebaDataset(data_loc + 'landmarks_train.csv', data_loc + 'attr_train.csv', data_loc + 'images',
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]), 
                        landmark_transform=transforms.Compose([
                            landmark_transforms.Rescale(224),
                            landmark_transforms.NormalizeLandmarks()
                        ]))
val_dataset = CelebaDataset(data_loc + 'landmarks_val.csv', data_loc + 'attr_val.csv', data_loc + 'images',
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]), 
                        landmark_transform=transforms.Compose([
                            landmark_transforms.Rescale(224),
                            landmark_transforms.NormalizeLandmarks()
                        ]))
test_dataset = CelebaDataset(data_loc + 'landmarks_test.csv', data_loc + 'attr_test.csv', data_loc + 'images',
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]), 
                        landmark_transform=transforms.Compose([
                            landmark_transforms.Rescale(224),
                            landmark_transforms.NormalizeLandmarks()
                        ]))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


num_classes = 10

# Define pretrained resnet model
resnet18 = models.resnet18(pretrained=True)
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, num_classes)
resnet18 = resnet18.to(device)

trainer_params = {
    'model': resnet18,
    'num_classes': 10,
    'train_loader': train_dataloader,
    'val_loader': val_dataloader,
    'test_loader': test_dataloader,
    'criterion': torch.nn.MSELoss,
    'criterion_args': {},
    'optimizer': torch.optim.Adam,
    'optimizer_args': {
         "lr": 1e-4,
         "weight_decay": 0
    },
    'scheduler': None,
    'scheduler_args': {},
    'debug': False,
}

trainer = Trainer(**trainer_params)

train_params = {
    'num_epochs': 10,
    'start_epochs': 0,
    'forward_args': {},
    'validate': True,
    'test': True,
    'save_dir': "../experiments/checkpoints",
    'tensorboard_dir': "../experiments/tensorboard",
    'log_dir': "../experiments/logs",
    'exp_name': 'celeba_baseline_resnet_1',
    'epoch_per_save': 1,
    'epoch_per_print': 1,
    'batch_per_print': 100,
    'batch_per_save': 100,
    'verbose': True
}

trainer.train(**train_params)