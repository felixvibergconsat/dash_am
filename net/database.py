#!/usr/bin/env python3

import numpy as np
from torch.utils.data import Dataset


class Database(Dataset):
    def __init__(self, type, source_transform=None, target_transform=None):
        self.source_transform = source_transform
        self.target_transform = target_transform

        if type == 'train':
            print('loading training data...', end=' ')
            self.target = np.load('data/target_train.npy', allow_pickle=True).astype('float64')
            self.source = np.load('data/source_train.npy', allow_pickle=True).astype('float64')
            print('done')
        elif type == 'evalu':
            print('loading validation data...', end=' ')
            self.target = np.load('data/target_valid.npy', allow_pickle=True).astype('float64')
            self.source = np.load('data/source_valid.npy', allow_pickle=True).astype('float64')
            print('done')

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        source = self.source[idx]
        target = self.target[idx]
        if self.source_transform and self.target_transform:
            source = self.source_transform(source)
            target = self.target_transform(target)

        return source, target
