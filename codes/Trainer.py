import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time

from Dataset import Poses3D


class Trainer:
    def __init__(self, network : nn.Module, training_datasets : list[Poses3D], validation_datasets : list[Poses3D], optimiser : torch.optim.Adam, 
                 scheduler : torch.optim.lr_scheduler.LinearLR=None, experiment_name='experiment', batch_size=32):
        self.network = network
        self.training_datasets = training_datasets
        self.training_dataloaders = [data.DataLoader(training_dataset, batch_size=1, shuffle=True) for training_dataset in self.training_datasets]
        self.validation_datasets = validation_datasets
        self.validation_dataloaders = [data.DataLoader(validation_dataset, batch_size=1) for validation_dataset in self.validation_datasets]
        
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.loss_function = self.get_loss_function()
        self.accuracy_function = self.get_accuracy_function()

        self.experiment_name = experiment_name
        self.writer = SummaryWriter(os.path.join('runs', experiment_name))
        self.batch_size = batch_size

    def get_loss_function(self) -> nn.Module:
        def loss(sample, target):
            return torch.mean(torch.sqrt(torch.sum(torch.square(sample - target), axis=1)))
        return loss #nn.MSELoss(reduction='sum')
    
    def get_accuracy_function(self) -> nn.Module:
        def acc(sample, target):
            return 1 - torch.mean(torch.sqrt(torch.sum(torch.square(sample - target), axis=1)))
        return acc

    def train(self, epochs):
        for e in range(epochs):
            print(f"Training: \nEpoch {e+1}\n-------------------------------")
            start = time()           
            tr_loss, tr_acc = self.train_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average training loss: {tr_loss}, Average training accuracy: {tr_acc}")
            print(f"Validation: \nEpoch {e+1}\n-------------------------------")
            start = time()       
            val_loss, val_acc = self.validation_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average validation loss: {val_loss}, Average validation accuracy: {val_acc}")
            self.log_loss(tr_loss, val_loss, e + 1)
            self.log_accuracy(tr_acc, val_acc, e + 1)

            self.save(e + 1)
        self.writer.flush()

        self.writer.close()

    def train_loop(self):
        self.network.train()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        for i, dataloader in enumerate(self.training_dataloaders):
            size = len(dataloader.dataset)
            
            for j, sample in enumerate(dataloader):     
                sequences = torch.squeeze(sample['sequences'], 0)
                s, c, h, w = sequences.shape
                targets  = torch.squeeze(sample['key_points'], 0)
                for k in range(0, s, self.batch_size):
                    sequence = sequences[k:k+self.batch_size]
                    key_points = targets[k:k+self.batch_size]
                    if len(sequence) < 2:
                        continue
                    samples_count += 1

                    prediction = self.network(sequence)
                    loss = self.loss(prediction, key_points)
                    self.optimise(loss)
                    acc = self.accuracy(prediction, key_points)

                    loss_overall += loss.item()
                    accuracy_overall += acc.item()
                if j % 10 == 0:
                    print(f'dataset: {dataloader.dataset.name}, {j}/{size}, loss: {loss.item()}, accuracy: {acc.item()}')

        if self.scheduler:
            self.scheduler.step()
        
        return loss_overall / samples_count, accuracy_overall / samples_count

    def validation_loop(self):
        self.network.eval()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        with torch.no_grad():
            for i, dataloader in enumerate(self.validation_dataloaders):
                size = len(dataloader.dataset)
                     
                for j, sample in enumerate(dataloader):     
                    sequences = torch.squeeze(sample['sequences'], 0)
                    s, c, h, w = sequences.shape
                    targets  = torch.squeeze(sample['key_points'], 0)
                    for k in range(0, s, self.batch_size):
                        sequence = sequences[k:k+self.batch_size]
                        key_points = targets[k:k+self.batch_size]
                        if len(sequence) == 0:
                            continue
                        samples_count += 1

                        prediction = self.network(sequence)
                        loss = self.loss(prediction, key_points)
                        acc = self.accuracy(prediction, key_points)

                        loss_overall += loss.item()
                        accuracy_overall += acc.item()
                    if j % 10 == 0:
                        print(f'dataset: {dataloader.dataset.name}, {j}/{size}, loss: {loss.item()}, accuracy: {acc.item()}')

        return loss_overall / samples_count, accuracy_overall / samples_count

    def loss(self, prediction : torch.Tensor, sample : dict[str, torch.Tensor]) -> torch.Tensor:
        j = self.network.joints
        if type(sample) is dict:
            ground_truth = torch.squeeze(sample['key_points'], 0)
        else:
            ground_truth = sample
        return self.loss_function(prediction, ground_truth)
    
    def accuracy(self, prediction : torch.Tensor, sample : dict[str, torch.Tensor]) -> torch.Tensor:
        j = self.network.joints
        if type(sample) is dict:
            ground_truth = torch.squeeze(sample['key_points'], 0)
        else:
            ground_truth = sample
        return self.accuracy_function(prediction, ground_truth)

    def optimise(self, loss : torch.Tensor):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def log_loss(self, training_loss, validation_loss, epoch):
        """self.writer.add_scalars('average loss:', 
                                {'training' : training_loss, 
                                 'validation' : validation_loss}, 
                                 epoch)"""
        
        self.writer.add_scalar('average training loss', training_loss, epoch)
        self.writer.add_scalar('average validation loss', validation_loss, epoch)
        
    def log_accuracy(self, training_accuracy, validation_accuracy, epoch):
        """self.writer.add_scalars('average accuracy:', 
                                {'training' : training_accuracy, 
                                 'validation' : validation_accuracy}, 
                                 epoch)"""
        
        self.writer.add_scalar('average training accuracy', training_accuracy, epoch)
        self.writer.add_scalar('average validation accuracy', validation_accuracy, epoch)
        
    def save(self, epoch):
        path = os.path.join('models', self.experiment_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        torch.save(self.network.state_dict(), os.path.join(path, f'net_{epoch}.pt'))
        torch.save(self.optimiser.state_dict(), os.path.join(path, f'optimiser_{epoch}.pt'))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(path, f'scheduler_{epoch}.pt'))