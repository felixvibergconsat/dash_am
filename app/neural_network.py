#!/usr/bin/env python3

from __future__ import print_function
from importlib import import_module
import torch
import numpy as np
import sys


class Model:
    def __init__(self, size):
        sys.path.insert(1, 'app/net')
        sys.path.insert(1, 'net')
        self.Net = import_module('Net').Net(size)
        self.net_args = self.Net.args()

        self.Net.load_state_dict(torch.load('app/net/net.pth', map_location='cpu'))

        use_cuda = not self.net_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.Net.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    def eval(self, numpy_source):
        self.Net.eval()
        source = torch.from_numpy(numpy_source).float()
        source = source.to(self.device)
        output = self.Net(source).squeeze()
        output = output.detach().cpu().numpy()
        return output
