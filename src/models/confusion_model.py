import os
import torch
from torch import nn, Tensor
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch.utils.tensorboard import SummaryWriter
from loss import total_loss, generator_loss, adversarial_loss
import torch.optim as optim
from typing import Dict, Iterable, Callable
import numpy as np

import sys
sys.path.append('../data')
sys.path.append('..')
sys.path.append('../utils')

import landmark_transforms
from trainer import Trainer

from torch.utils.data import sampler


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def generator(seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model = None
    return model

def adversary(seed=None):
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


def adversary_classifier(layer_size, num_attr):
    """
    Predicts all sensitive attribute scores given feature representations
    """
    model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(layer_size, 256),
      nn.LeakyReLU(0.01),
      nn.Linear(256, 256),
      nn.LeakyReLU(0.01),
      nn.Linear(256, num_attr)
    )

    return model


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


def run_model(generator, adversary, feature_extractor, g_solver, a_solver,
              train_loader, val_loader, w=10, eps=2, alpha=1, print_every=100, num_epochs=20, verbose=True, num_classes=10, num_attributes=6, num_adversary_repetitions=1,
              save_dir = "experiments/checkpoints",
              tensorboard_dir = "experiments/tensorboard",
              log_dir = "experiments/logs",
              figure_dir = "experiments/figures",
              exp_name = 'experiment'):
    """
    Inputs:
    - generator, adversary: PyTorch models for the generator, adversary, predictor
    - feature_extractor: extracts feature representation created by generator
    - g_solver, a_solver: torch.optim Optimizers for training the generator/adversary
    - train_loader, val_loader: DataLoader objects containing image, landmark, and attributes
    - print_every: print loss after 'print_every' iterations
    """
    # log file for saving
    def _print_log(msg, log_file=None, verbose=True):
        if log_file is not None:
            with open(log_file, 'a') as file:
                print(msg, file=file)
        print(msg) if verbose else 0
        
    # Plot loss graph function
    def loss_graph(figure_file, train, val, show=False):
        plt.close()
        fig = plt.figure()
        
        plt.title(f"Loss, {len(train)} epochs")
        
        plt.plot(train, 'b-') if train else 0
        plt.plot(val, 'g-') if val else 0
        
        plt.savefig(figure_file)
        plt.savefig(sys.stdout.buffer) if show else 0
    
    for folder in [save_dir, tensorboard_dir, log_dir, figure_dir]:
        if folder is not None and not os.path.exists(folder):
            os.makedirs(folder)
            
    experiment_name = f"{exp_name}"
    save_dir = os.path.join(save_dir, experiment_name) if save_dir is not None else None
    tensorboard_dir = os.path.join(tensorboard_dir, experiment_name) if tensorboard_dir is not None else None
    figure_dir = os.path.join(figure_dir, experiment_name) if figure_dir is not None else None
        
    # Create subfolders if not existent
    for folder in [save_dir, tensorboard_dir, figure_dir]:
        if folder is not None and not os.path.exists(folder):
            os.makedirs(folder)
                
    for folder in [save_dir, tensorboard_dir, log_dir, figure_dir]:
        if folder is not None:
            assert(os.path.exists(folder))
        
    writer = SummaryWriter(tensorboard_dir) if tensorboard_dir is not None else None
    log_file = os.path.join(log_dir, experiment_name) if log_dir is not None else None
        
    _print_log(f"M Experiment {exp_name}", log_file=log_file, verbose=verbose)
    _print_log(f"M Log path: {log_file}", log_file=log_file, verbose=verbose)
    _print_log(f"M Figure path: {figure_dir}", log_file=log_file, verbose=verbose)
    _print_log(f"M Model save directory: {save_dir}", log_file=log_file, verbose=verbose)
    _print_log(f"M Tensorboard directory: {tensorboard_dir}", log_file=log_file, verbose=verbose)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    generator = generator.to(device=device)
    adversary = adversary.to(device=device)
    feature_extractor = feature_extractor.to(device=device)
        
    #############################
    #         Train Loop        #
    #############################
        
    _print_log("O ----------==========Training Loop==========----------", log_file=log_file, verbose=verbose)
    images = []
    iter_count = 0
    gen_loss_train_hist = []
    gen_loss_val_hist = []
    adv_loss_train_hist = []
    adv_loss_val_hist = []
    
    for epoch in range(num_epochs):
        generator.train()
        feature_extractor.train()
        adversary.train()
        
        gen_losses = []
        adv_losses = []
        for batch_idx, batch in enumerate(train_loader, 0):
            inputs = batch['image'].to(device=device)
            labels = batch['landmarks'].view(-1, num_classes).to(device=device)
            target_attr = batch['attributes'].view(-1, num_attributes).float().to(device=device)
            torch.autograd.set_detect_anomaly(True)
            # zero gradient
            g_solver.zero_grad()
            a_solver.zero_grad()

            # generator: create feature representations and predicts landmark locations using feature representations
            output = generator(inputs)
                
            # Extract base features
            with torch.no_grad():
                feat_repr = list(feature_extractor(inputs).items())[0][1] 
                
            output_attr = adversary(feat_repr)
            
            # calculator generator loss and update
            g_loss = generator_loss(output, labels, output_attr, target_attr, w, eps, alpha)
            g_loss.backward(retain_graph=True)
            g_solver.step()
            
            for _ in range(num_adversary_repetitions - 1):
                # adversary: predict sensitive attributes using feature representations
                # calculator adversarial loss and update
                a_loss = adversarial_loss(output_attr, target_attr)
                a_loss.backward(retain_graph=True)
                a_solver.step()
                output_attr = adversary(feat_repr)
            a_loss = adversarial_loss(output_attr, target_attr)
            a_loss.backward()
            a_solver.step()
            
            
            gen_losses.append(g_loss.item())
            adv_losses.append(a_loss.item())

            if verbose and iter_count % print_every == 0:
                _print_log('B Iter: {}, G Loss: {:.4}, A Loss:{:.4}'.format(iter_count, g_loss.item(), a_loss.item()), log_file=log_file, verbose=verbose)

            iter_count += 1
        gen_loss = sum(gen_losses) / len(gen_losses)
        adv_loss = sum(adv_losses) / len(adv_losses)
        gen_loss_train_hist.append(gen_loss)
        writer.add_scalar('loss/train', gen_loss, epoch) if writer is not None else 0
        adv_loss_train_hist.append(adv_loss)
        writer.add_scalar('loss/adv_train', adv_loss, epoch) if writer is not None else 0
        
        _print_log(f'T Epoch {epoch} loss: {gen_loss}, adversary loss: {adv_loss}', log_file=log_file, verbose=verbose)
        
        generator.eval()
        feature_extractor.eval()
        adversary.eval()
        
        gen_losses = []
        adv_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, 0):
                inputs = batch['image'].to(device=device)
                labels = batch['landmarks'].view(-1, num_classes).to(device=device)
                target_attr = batch['attributes'].view(-1, num_attributes).float().to(device=device)

                # generator: create feature representations and predicts landmark locations using feature representations
                output = generator(inputs)

                with torch.no_grad():
                    feat_repr = list(feature_extractor(inputs).items())[0][1] #TODO

                # adversary: predict sensitive attributes using feature representations
                output_attr = adversary(feat_repr)

                # calculator generator loss and update
                g_loss = generator_loss(output, labels, output_attr, target_attr, w, eps, alpha)

                # calculator adversarial loss and update
                a_loss = adversarial_loss(output_attr, target_attr)

                gen_losses.append(g_loss.item())
                adv_losses.append(a_loss.item())
        gen_loss = sum(gen_losses) / len(gen_losses)
        adv_loss = sum(adv_losses) / len(adv_losses)
        gen_loss_val_hist.append(gen_loss)
        writer.add_scalar('loss/val', gen_loss, epoch) if writer is not None else 0
        adv_loss_val_hist.append(adv_loss)
        writer.add_scalar('loss/adv_val', adv_loss, epoch) if writer is not None else 0
        _print_log(f'V Epoch {epoch} loss: {gen_loss}, adversary loss: {adv_loss}', log_file=log_file, verbose=verbose)
        
        if save_dir is not None:
            save_folder = os.path.join(save_dir, "checkpoint_" + str(epoch))
            os.makedirs(save_folder) if not os.path.exists(save_folder) else 0
            torch.save(generator.state_dict(), os.path.join(save_folder,"model.pt"))
            torch.save(adversary.state_dict(), os.path.join(save_folder,"adversary.pt"))
            _print_log(f"S Saved checkpoint at {save_folder}", log_file=log_file, verbose=verbose)
            
        if figure_dir is not None:
            train_figure_file = os.path.join(figure_dir, "loss_train.png")
            loss_graph(train_figure_file, train=gen_loss_train_hist, val=gen_loss_val_hist, show=False)
            adv_figure_file = os.path.join(figure_dir, "loss_train_adv.png")
            loss_graph(adv_figure_file, train=adv_loss_train_hist, val=adv_loss_val_hist, show=False)
            _print_log(f"S Saved train loss figure at {train_figure_file}", log_file=log_file, verbose=verbose)
        

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

        
