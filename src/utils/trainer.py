import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot
from torchinfo import summary
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os

class Trainer:
    """
    Trainer object to handle training and testing based on input model, dataloaders, criteria, 
    optimizers. 
    
    Log details: Lines classified by first character
        M: Metadata
        T: Training loss (epoch)
        B: Training loss (batch)
        V: Validation loss (epoch)
        W: Best validation loss so far
        R: Training results (only if val set provided)
        S: Saving information
        O: Other
    """
    def __init__(self,
                 model,
                 num_classes,
                 train_loader = None,
                 val_loader = None,
                 test_loader = None,
                 criterion = torch.nn.MSELoss,
                 criterion_args = {},
                 optimizer = torch.optim.Adam,
                 optimizer_args = {
                     "lr": 1e-4,
                     "weight_decay": 0
                 },
                 scheduler = None,
                 scheduler_args = {},
                 scheduler_step_val = False,
                 debug = False,
                 is_preset_criterion = True):
        """
        Arguments:
            model: model to train
            num_classes: num classes to train to (required)
            train_loader: Training dataloader (required to train model)
            val_loader: Validation dataloader (optional)
            test_loader: Test dataloader (optional)
            criterion: Loss function class, must take in label predictions and actual labels (required to train model)
            criterion_args: Args for loss function initialization (optional)
            optimizer: Optimizer class (required to train model)
            optimizer_args: Additional args for optimizer initialization (optional)
            scheduler: Scheduler class (not yet implemented),
            scheduler_args: Additional args for scheduler initialization (not yet implemented),
            scheduler_step_val: whether to pass in val loss to scheduler step function,
            device: Device str to run trainer on
            debug: Print statements during initialization
        """
        
        self.debug = debug
        print(model) if self.debug else 0
        
        if model is None:
            self.model = model
            return
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        print(f"Running on device: {device}") if self.debug else 0
        
        self.model = model.to(device=self.device)
        self.num_classes = num_classes
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        if is_preset_criterion:
            self.criterion = criterion(**criterion_args)
        else:
            self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), **optimizer_args)
        self.scheduler = scheduler(self.optimizer, **scheduler_args) if scheduler is not None else None
        self.scheduler_step_val = scheduler_step_val
        
        # Will store train/val loss for each epoch
        self.train_loss_hist = []
        self.val_loss_hist = []
        
        # Store information about epoch with lowest validation 
        self.best_epoch = -1
        self.best_model_dict = self.model.state_dict(),
        self.best_val = np.inf
        self.best_train = 0
       
    # Write message to log file, stdout, or both
    def _print_log(self, msg, log_file=None, verbose=True):
        if log_file is not None:
            with open(log_file, 'a') as file:
                print(msg, file=file)
        print(msg) if verbose else 0
    
    def train(self,
              num_epochs,
              start_epochs = 0,
              forward_args = {},
              validate = True,
              test = True,
              val_criterion = None,
              save_dir = "experiments/checkpoints",
              tensorboard_dir = "experiments/tensorboard",
              log_dir = "experiments/logs",
              figure_dir = "experiments/figures",
              exp_name = 'experiment',
              epoch_per_save = 1,
              batch_per_save = 1000000,
              epoch_per_print = 1,
              batch_per_print = 100,
              verbose = True):
        """
        Arguments:
            num_epochs: Number of epochs to train (required)
            start_epochs: Starting epoch count (optional)
            forward_args: Additional args for model forward (optional)
            validate: whether to validate at each epoch and keep track of best model if val dataloader available
            test: whether to test on the final model if test dataloader available
            val_criterion: criterion to validate and test on, defaults to trainer's criterion
            save_dir: directory to save model checkpoints in, None to not save
            tensorboard_dir: directory to save tensorboard in, None to not save
            log_dir: directory to save log file in, None to not save
            figure_dir: directory to save loss figures in, None to not save
            exp_name: identifier of experiment for saving purposes
            epoch_per_save: Number of epochs between consecutive model checkpoints
            batch_per_save: Number of batches between consecutive model checkpoints 
                WARNING: Making this low will lead to massive memory usage. Default is
                set to high number.
            epoch_per_print: Number of epochs between consecutive train/val prints
            batch_per_print: Number of batches between consecutive train prints
            verbose: Whether to print log to stdout
        """
        
        assert self.train_loader is not None
        val_criterion = self.criterion if val_criterion is None else val_criterion
        
        # Create experiment folders if not existent
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
        
        self._print_log(f"M Experiment {exp_name}", log_file=log_file, verbose=verbose)
        self._print_log(f"M Log path: {log_file}", log_file=log_file, verbose=verbose)
        self._print_log(f"M Figure path: {figure_dir}", log_file=log_file, verbose=verbose)
        self._print_log(f"M Model save directory: {save_dir}", log_file=log_file, verbose=verbose)
        self._print_log(f"M Tensorboard directory: {tensorboard_dir}", log_file=log_file, verbose=verbose)

        #############################
        #         Train Loop        #
        #############################
        
        self._print_log("O ----------==========Training Loop==========----------", log_file=log_file, verbose=verbose)
        end_epochs = start_epochs + num_epochs
        start_batches = len(self.train_loader) * start_epochs
        end_batches = len(self.train_loader) * end_epochs
        batches = start_batches
        for epoch in range(start_epochs, end_epochs):
            # Training step and save training loss
            self.model.train()
            epoch_losses = []
            for batch_idx, batch in enumerate(self.train_loader, 0):
                inputs = batch['image'].to(device=self.device)
                labels = batch['landmarks'].view(-1, self.num_classes).to(device=self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs, **forward_args)  
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                
                if batch_idx % batch_per_print == 0:
                    self._print_log(f'B Epoch {epoch} batch {batch_idx} loss: {epoch_losses[-1]}', log_file=log_file, verbose=verbose)
                
                if save_dir is not None and batches % batch_per_save == 0:
                    save_folder = os.path.join(save_dir, "checkpoint_batch_" + str(batches))
                    os.makedirs(save_folder) if not os.path.exists(save_folder) else 0
                    torch.save(self.model, os.path.join(save_folder,"model.pt"))
                    torch.save(self.optimizer, os.path.join(save_folder,"optimizer.pt"))
                    self._print_log(f"S Saved checkpoint at {save_folder}", log_file=log_file, verbose=verbose)
                batches += 1
            
            train_loss = sum(epoch_losses) / len(epoch_losses)
            self.train_loss_hist.append(train_loss)
            writer.add_scalar('loss/train', train_loss, epoch) if writer is not None else 0
            if epoch % epoch_per_print == 0:
                self._print_log(f'T Epoch {epoch} loss: {train_loss}', log_file=log_file, verbose=verbose)
            
            if validate and self.val_loader is not None:
                # Evaluate and save validation loss
                val_loss = self.test(test_loader = self.val_loader,
                                     criterion = val_criterion,
                                     forward_args = forward_args)
                self.val_loss_hist.append(val_loss)
                writer.add_scalar('loss/val', val_loss, epoch)
                if epoch % epoch_per_print == 0:
                    self._print_log(f'V Epoch {epoch} loss: {val_loss}', log_file=log_file, verbose=verbose)
                 
                if self.scheduler is not None:
                    self.scheduler.step(val_loss) if self.scheduler_step_val else self.scheduler.step()
                
                # Update info about epoch with best validation loss
                if val_loss < self.best_val:
                    self._print_log(f'W Epoch {epoch} val {val_loss} beats epoch {self.best_epoch} val {self.best_val}', log_file=log_file, verbose=verbose)
                    self.best_epoch = epoch
                    self.best_model = self.model.state_dict()
                    self.best_val = val_loss
                    self.best_train = train_loss
                
            # Save checkpoint state dicts
            if save_dir is not None and epoch % epoch_per_save == 0:
                save_folder = os.path.join(save_dir, "checkpoint_" + str(epoch))
                os.makedirs(save_folder) if not os.path.exists(save_folder) else 0
                torch.save(self.model, os.path.join(save_folder,"model.pt"))
                torch.save(self.optimizer, os.path.join(save_folder,"optimizer.pt"))
                self._print_log(f"S Saved checkpoint at {save_folder}", log_file=log_file, verbose=verbose)
            
            if figure_dir is not None and epoch % epoch_per_save == 0:
                train_figure_file = os.path.join(figure_dir, "loss_train.png")
                self.loss_graph(train_figure_file, train=True, val=False, show=False)
                self._print_log(f"S Saved train loss figure at {train_figure_file}", log_file=log_file, verbose=verbose)
                if validate and self.val_loader is not None:
                    val_figure_file = os.path.join(figure_dir, "loss_val.png")
                    self.loss_graph(val_figure_file, train=False, val=True, show=False)
                    self._print_log(f"S Saved val loss figure at {val_figure_file}", log_file=log_file, verbose=verbose)
           
        if validate and self.val_loader is not None:
            self._print_log("O ----------==========Best Val Results==========----------", log_file=log_file, verbose=verbose)
            self._print_log(f"R Best epoch: {self.best_epoch}", log_file=log_file, verbose=verbose)
            self._print_log(f"R Best val train loss: {self.best_train}", log_file=log_file, verbose=verbose)
            self._print_log(f"R Best val loss: {self.best_val}", log_file=log_file, verbose=verbose)
        
        self._print_log("O ----------==========Final Model Results==========----------", log_file=log_file, verbose=verbose)
        self._print_log(f"R End epoch: {end_epochs - 1}", log_file=log_file, verbose=verbose)
        self._print_log(f"R End train loss: {self.train_loss_hist[-1]}", log_file=log_file, verbose=verbose)
        if validate and self.val_loader is not None:
            self._print_log(f"R End val loss: {self.val_loss_hist[-1]}", log_file=log_file, verbose=verbose)
        if test and self.test_loader is not None:
            test_loss = self.test(test_loader = self.test_loader,
                                  criterion = val_criterion,
                                  forward_args = forward_args)
            self._print_log(f"R End test loss: {test_loss}", log_file=log_file, verbose=verbose)
        
    def test(self,
             model = None,
             test_loader = None,
             criterion = None,
             forward_args = {},
             best_model = False):
        """
        Arguments:
            model: Model to test, defaults to model stored in Trainer instance
            test_loader: Dataloader to evaluate on, defaults to test dataloader stored in instance
            criterion: Criterion to evaluate on, defaults to criterion provided at initialization
            forward_args: Additional args to provide to model forward
            best_model: Whether to use model with best val loss during training (not yet implemented)
        """
        model = model if model is not None else self.model
        test_loader = self.test_loader if test_loader is None else test_loader
        criterion = self.criterion if criterion is None else criterion
        self.model.eval()
        test_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs = batch['image'].to(device=self.device)
                labels = batch['landmarks'].view(-1, self.num_classes).to(device=self.device)

                outputs = self.model(inputs, **forward_args)  # (batch_size, length, num_classes)
                
                loss = criterion(outputs, labels).item()
                test_losses.append(loss)
        test_loss = sum(test_losses) / len(test_losses)
        return test_loss
    
    def loss_graph(self, figure_file, train=True, val=True, show=False):
        plt.close()
        fig = plt.figure()
        
        plt.title(f"Loss, {len(self.train_loss_hist)} epochs")
        
        plt.plot(self.train_loss_hist, 'b-') if train else 0
        plt.plot(self.val_loss_hist, 'g-') if val else 0
        
        plt.savefig(figure_file)
        plt.savefig(sys.stdout.buffer) if show else 0
