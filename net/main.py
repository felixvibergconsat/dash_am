#!/usr/bin/env python3

from __future__ import print_function
from datetime import datetime, timedelta
from importlib import import_module
from database import Database

import torch
import csv

import torch.optim as optim
import pandas as pd
import numpy as np


def start_session():
    with open('loss_over_epochs_ex.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(
            ['started', 'started', 'started', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


class Model:
    def __init__(self):
        self.best = 100
        self.Net = import_module('Net').Net()
        self.net_args = self.Net.args()

        #self.Net.load_state_dict(torch.load('net.pth', map_location='cpu'))

        use_cuda = not self.net_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.Net.to(self.device)
        self.kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
        self.lr = self.net_args.lr * 0.2
        self.optimizer = optim.Adam(self.Net.parameters(), lr=self.lr, betas=(0.8, 0.99),
                                    eps=1e-5)
        self.criterion = torch.nn.SmoothL1Loss()
        self.epoch = 0

        source_transform = self.Net.source_transform()
        target_transform = self.Net.target_transform()

        train_db = Database('train', source_transform=source_transform, target_transform=target_transform)
        evalu_db = Database('evalu', source_transform=source_transform, target_transform=target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.net_args.batch_size,
                                                        shuffle=True, **self.kwargs)
        self.evalu_loader = torch.utils.data.DataLoader(evalu_db, batch_size=self.net_args.batch_size,
                                                        shuffle=True, **self.kwargs)
        self.loss_epoch, self.loss_epoch_evalu = 0, 0

    def train(self, helpers):
        torch.cuda.empty_cache()
        self.Net.train()
        loss_log = 0
        self.loss_epoch = 0

        for batch_idx, (source, target) in enumerate(self.train_loader):
            source, target = source.to(self.device), target.to(self.device)
            output = self.Net(source)
            loss = self.criterion(target, output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_log += loss
            self.loss_epoch += loss.item()

            if batch_idx % self.net_args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.6f}\tLoss(last): {:.6f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss_log.item(), loss.item()))
        self.loss_epoch /= len(self.train_loader)

    def evaluate(self, helpers):
        self.Net.eval()
        loss_log = 0
        self.loss_epoch_evalu = 0

        with torch.no_grad():
            torch.cuda.empty_cache()
            for batch_idx, (source, target) in enumerate(self.evalu_loader):

                source, target = source.to(self.device), target.to(self.device)
                output = self.Net(source)
                loss = self.criterion(target, output)
                loss_log += loss
                self.loss_epoch_evalu += loss.item()

                if batch_idx % self.net_args.log_interval == 0:
                    print('Validation Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.6f}\tLoss(last): {:.6f}'.format(
                        self.epoch + 1, batch_idx * len(source), len(self.evalu_loader.dataset),
                        100. * batch_idx / len(self.evalu_loader), loss_log.item(), loss.item()))
        self.loss_epoch_evalu /= len(self.evalu_loader)

        if self.loss_epoch_evalu < self.best:
            self.best = self.loss_epoch_evalu
            self.save_model()
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr']*0.95

    def save_model(self):
        print(' ' + '-' * 64, '\nSaving\n', '-' * 64)
        model_full_path = '../app/net/net.pth'
        torch.save(self.Net.state_dict(), model_full_path)

    def loss_to_file(self):
        with open('loss_over_epochs_ex.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.loss_epoch, self.loss_epoch_evalu,
                             datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def main():
    helpers = np.load('../app/data/helpers.npy', allow_pickle=True)
    model = Model()
    start_session()

    for epoch in range(model.epoch, model.net_args.epochs + 1):
        model.train(helpers)
        model.evaluate(helpers)
        model.epoch = epoch + 1
        model.loss_to_file()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main()
