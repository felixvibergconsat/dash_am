import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import argparse
import custom_transforms

class Net(nn.Module):
    def __init__(self, size):
        super(Net, self).__init__() 

        self.fc1 = nn.Linear(size, 42)
        self.fc2 = nn.Linear(42, 42)
        self.fc3 = nn.Linear(42, 42)
        self.fc4 = nn.Linear(42, 42)
        self.fc5 = nn.Linear(42, 42)
        self.fc6 = nn.Linear(42, 42)
        self.fc7 = nn.Linear(42, 42)
        self.fc8 = nn.Linear(42, 42)
        self.fc9 = nn.Linear(42, 42)
        self.fc10 = nn.Linear(42, 1)

        self.do1 = nn.Dropout(0.1)
        self.do2 = nn.Dropout(0.1)
        self.do3 = nn.Dropout(0.1)
        self.do4 = nn.Dropout(0.1)
        self.do5 = nn.Dropout(0.1)
        self.do6 = nn.Dropout(0.1)
        self.do7 = nn.Dropout(0.1)
        self.do8 = nn.Dropout(0.1)
        self.do9 = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        self.do1(x)
        x = torch.relu(self.fc2(x))
        self.do2(x)
        x = torch.relu(self.fc3(x))
        self.do3(x)
        x = torch.relu(self.fc4(x))
        self.do8(x)
        x = torch.relu(self.fc9(x))
        self.do9(x)
        x = self.fc10(x)
        return torch.tanh(x)

    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch')
        parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=60, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.005)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--train_db_path', type=str, default='data.csv',
                            help='path to train csv')
        parser.add_argument('--val_db_path', type=str, default='data.csv',
                            help='path to val csv')
        parser.add_argument('--log_interval', type=int, default=10,
                            help='log interval')
        parser.add_argument('--loss_function', type=str, default='mse_loss',
                            help='the loss function to use. Must be EXACTLY as the function is called in pytorch docs')
        self.args, unknown = parser.parse_known_args()
        return self.args

    @staticmethod
    def source_transform():
        source_transform = transforms.Compose([custom_transforms.ToTensor()])
        return source_transform

    @staticmethod
    def target_transform():
        target_transform = transforms.Compose([custom_transforms.ToTensor()])
        return target_transform
