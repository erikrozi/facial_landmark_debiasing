# Adversarial Domain Adaptation Loss

import torch
import torch.nn as nn
import math


def total_loss_per_sample(output, target, feat_repr, target_attr, sens_attr, w=10, eps=2, alpha=1):
    '''
    Calculates the join-loss for adversarial domain adaptation

    PARAMETERS
    output: output labels predicted by model
    target: true labels
    feat_repr: feature representations produced by the classifier
    target_attr: true labels of all attributes
    sens_attr: list of ints representing sensitive attributes

    HYPERPARAMETERS
    w: wing loss hyperparameter. sets the range of the non-linear part to be (-w, w). default 10
    eps: wing loss hyperparameter. controls curvature of the non-linear region. default 2
    alpha: controls influence of confusion loss. defualt 1
    '''
    primary_loss = wing_loss(output, target, w, eps)
    adversary_loss = adversarial_loss(feat_repr, target_attr, sens_attr)
    confusion_loss = domain_confusion_loss(feat_repr, target_attr, sens_attr)

    loss = primary_loss + adversary_loss + alpha * confusion_loss
    return loss


def wing_loss(output, target, w=10, eps=2):
    '''
    Calculate average wing loss over all samples

    output = output labels predicted by model
    target = true labels
    w = sets the range of the non-linear part to be (-w, w) (hyperparameter)
    eps = controls curvature of the non-linear region (hyperparameter)

    Default w, eps chosen to be the one presented in Feng et al., 2018 (https://arxiv.org/pdf/1711.06753.pdf)
    '''
    num_samples = output.shape[0]
    abs_diff = torch.abs(output - target)
    C = w - w * math.log(1 + w / eps)

    loss_per_sample_feature = torch.where(abs_diff < w, w * torch.ln(1 + abs_diff / eps), abs_diff - C)
    loss = 1. / num_samples * torch.sum(loss_per_sample_feature)

    return loss


def adversarial_loss(feat_repr, target_attr, sens_attr):
    '''
    Calculates the adversarial loss of each sensitive attribute and returns the sum

    feat_repr: feature representations produced by the classifier
    target: true labels of all attributes
    sens_attr: list of ints representing sensitive attributes
    '''
    softmax = nn.Softmax(dim=1) # check dim
    probs = softmax(feat_repr)

    # only get entries corresponding sensitive attributes, and compute loss
    # need to check dimensions
    sens_probs = probs[sens_attr]
    sens_target = target_attr[sens_attr]
    loss = sens_probs[:, sens_target].sum()
    return loss


def domain_confusion_loss(feat_repr, target_attr, sens_attr):
    '''
    Calculates the domain confusion loss of each sensitive attribute and returns the sum

    feat_repr: feature representations produced by the classifier
    target: true labels of all attributes
    sens_attr: list of ints representing sensitive attributes
    '''
    cross_entropy = nn.CrossEntropyLoss()

    # only get entries corresponding sensitive attributes, and compute loss
    sens_feat_repr = feat_repr[sens_attr]
    sens_target = target_attr[sens_attr]
    loss_per_attr = cross_entropy(sens_feat_repr, sens_target)
    loss = loss_per_attr.sum()

    return loss