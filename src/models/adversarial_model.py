import os
import torch
from torch import nn
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch.utils.tensorboard import SummaryWriter
from loss import total_loss, generator_loss, adversarial_loss
import torch.optim as optim
import numpy as np

import sys
sys.path.append('../data')
sys.path.append('..')
sys.path.append('../utils')

import landmark_transforms
from trainer import Trainer

from torch.utils.data import sampler

NOISE_DIM = 96

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def discriminator(seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model = None
    return model


def get_optimizer(model, lr=1e-3, beta1=0.5, beta2=0.999):
    """
    Constructs and returns An Adam optimizer for the model with the desired hyperparameters
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizer


def discriminator_classifier(batch_size, num_attr):
    """
    Predicts all sensitive attribute scores given feature representations
    """
    model = nn.Sequential(
        nn.Conv2d(1, 32, 5, 1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 5, 1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, 2),
        Flatten(),
        nn.Linear(4 * 4 * 64, 4 * 4 * 64),
        nn.LeakyReLU(0.01),
        nn.Linear(4 * 4 * 64, num_attr)
    )

    return model


# TODO: code the generator. add a forward hook to the model to save feat reprs (called feature_representations)
def generator_model():
    return

def run_a_gan(generator, adversary, feature_extractor, g_solver, a_solver, p_solver, generator_loss, adversarial_loss,
              loader_train, w=10, eps=2, alpha=1, print_every=250, batch_size=128, num_epochs=10, verbose=True):
    """
    Train a GAN!

    Inputs:
    - generator, adversary: PyTorch models for the generator, adversary, predictor
    - feature_extractor: extracts feature representation created by generator
    - g_solver, a_solver: torch.optim Optimizers for training the generator/adversary
    - generator_loss, adversarial_loss: compute generator and adversary loss
    - verbose:
    - print_every: print loss after 'print_every' iterations
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    images = []
    iter_count = 0
    for epoch in range(num_epochs):
        for data, target in loader_train:
            if len(data) != batch_size:
                continue

            # zero gradient
            g_solver.zero_grad()
            a_solver.zero_grad()

            # generator: create feature representations and predicts landmark locations using feature representations
            output = generator(data)
            feat_repr = generator.feature_representations()

            # adversary: predict sensitive attributes using feature representations
            output_attr = adversary(feat_repr)
            target_attr = data

            # calculator generator loss and update
            g_loss = generator_loss(output, target, output_attr, w, eps, alpha)
            g_loss.backward()
            g_solver.step()

            # calculator adversarial loss and update
            a_loss = adversarial_loss(output_attr, target_attr)
            a_loss.backward()
            a_solver.step()

            if verbose and iter_count % print_every == 0:
                print('Iter: {}, G Loss: {:.4}, A Loss:{:.4}'.format(iter_count, g_loss.item(), a_loss.item()))

            iter_count += 1

    return


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

