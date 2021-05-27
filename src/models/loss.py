"""
Adversarial Domain Adaptation Loss

Use the total_loss() function:
N = number of samples
L = number of landmarks
K = number of sensitive attributes

PARAMETERS
- output: (N, 2L) tensor of landmark locations predicted by model (continuous)
- target: (N, 2L) tensor of true landmark locations (continuous)
- output_attr: (N, K, 2) tensor of predicted scores of sensitive attributes, given feature representations
- target_attr: (N, K) tensor of true sensitive attribute labels. all entries are 0, 1

HYPERPARAMETERS
- w: wing loss hyperparameter. sets the range of the non-linear part to be (-w, w). default 10
- eps: wing loss hyperparameter. controls curvature of the non-linear region. default 2
- alpha: controls influence of confusion loss. default 1
- Default w, eps chosen to be the one presented in Feng et al., 2018 (https://arxiv.org/pdf/1711.06753.pdf)
"""


import torch
import torch.nn as nn
import math


def total_loss(output, target, output_attr, target_attr, w=10, eps=2, alpha=1):
    """
    Calculates the total loss for adversarial domain adaptation
    """
    primary_loss = wing_loss(output, target, w, eps)
    adversary_loss = adversarial_loss(output_attr, target_attr)
    confusion_loss = domain_confusion_loss(output_attr)

    loss = primary_loss + adversary_loss + alpha * confusion_loss
    return loss


def wing_loss(output, target, w=10, eps=2):
    """
    Calculates average wing loss of all sample output/target differences
    Calculates average wing loss of each landmark location
    """
    abs_diff = torch.abs(output - target)
    C = w - w * math.log(1 + w / eps)

    loss_per_sample_feature = torch.where(abs_diff < w, w * torch.ln(1 + abs_diff / eps), abs_diff - C)
    loss_per_sample = loss_per_sample_feature.sum(dim=-1)
    # loss_per_feature = loss_per_sample_feature.mean(dim=0)
    loss = loss_per_sample.mean()

    return loss


def generator_loss(output, target, output_attr, w=10, eps=2, alpha=1):
    """
    Calculates the generator loss
    """
    primary_loss = wing_loss(output, target, w, eps)
    confusion_loss = domain_confusion_loss(output_attr)

    loss = primary_loss + alpha * confusion_loss
    return loss


def adversarial_loss(output_attr, target_attr):
    """
    Calculates the adversarial loss of each sensitive attribute and returns the sum
    """
    # compute log-softmax of each sensitive attribute for each sample
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(output_attr)

    # total loss
    loss = log_probs[:, :, target_attr].sum()
    return loss


def domain_confusion_loss(output_attr):
    """
    Calculates the domain confusion loss of each sensitive attribute and returns the sum
    """
    # compute log-softmax of each sensitive attribute for each sample
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(output_attr)

    # total loss
    loss = log_probs.mean()
    return loss
