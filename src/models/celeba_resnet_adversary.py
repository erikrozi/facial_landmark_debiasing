import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch import nn
from loss import wing_loss, generator_loss, adversarial_loss

import sys
sys.path.append('../data')
sys.path.append('..')
sys.path.append('../utils')

from datasets import WFLWDataset, CelebaDataset
import landmark_transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer
from adversarial_model import run_model, FeatureExtractor, adversary_classifier, get_optimizer

data_loc = '/home/data/celeba/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EXPERIMENT_NAME = 'celeba_resnet_adversary_2'

train_dataset = CelebaDataset(data_loc + 'landmarks_train.csv', data_loc + 'attr_train.csv', data_loc + 'images/img_celeba',
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]), 
                        landmark_transform=transforms.Compose([
                            landmark_transforms.Rescale(224),
                            landmark_transforms.RandomRotation(20),
                            landmark_transforms.NormalizeLandmarks()
                        ]))
val_dataset = CelebaDataset(data_loc + 'landmarks_val.csv', data_loc + 'attr_val.csv', data_loc + 'images/img_celeba',
                        transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]), 
                        landmark_transform=transforms.Compose([
                            landmark_transforms.Rescale(224),
                            landmark_transforms.NormalizeLandmarks()
                        ]))
test_dataset = CelebaDataset(data_loc + 'landmarks_test.csv', data_loc + 'attr_test.csv', data_loc + 'images/img_celeba',
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
num_attributes = 40

# Define pretrained resnet model
resnet18 = models.resnet18(pretrained=False)
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, num_classes)
resnet18 = resnet18.to(device)

resnet_features = FeatureExtractor(resnet18, layers=["avgpool"])
adversary = adversary_classifier(layer_size=512, num_attr=num_attributes)

g_solver = get_optimizer(resnet18, lr=1e-4)
a_solver = get_optimizer(adversary, lr=1e-3)

trainer_params = {
    'generator': resnet18,
    'adversary': adversary,
    'feature_extractor': resnet_features,
    'g_solver': g_solver,
    'a_solver': a_solver,
    'train_loader': train_dataloader,
    'val_loader': val_dataloader,
    'num_adversary_repetitions': 5,
    'w': 10,
    'eps': 2,
    'alpha': 6,
    'print_every': 100,
    'num_epochs': 25,
    'num_classes': num_classes,
    'num_attributes': num_attributes,
    'save_dir': "../experiments/checkpoints",
    'tensorboard_dir': "../experiments/tensorboard",
    'log_dir': "../experiments/logs",
    'exp_name': EXPERIMENT_NAME,
}

run_model(**trainer_params)
