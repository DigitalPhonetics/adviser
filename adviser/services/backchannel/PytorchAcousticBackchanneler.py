###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

import torch
from torch import nn
import numpy as np

class PytorchAcousticBackchanneler(nn.Module):
    """Class for defining the Deep Backchannel model in PyTorch"""

    def __init__(self, parameters:list=[], load_params:bool=False):
        """
        Defines the elements/layers of the neural network as well as loads the pretrained parameters

        The model is constituted by two parallel CNNs followed by a concatenation, a  FFN and a softmax layer.

        Args:
            parameters (list): list of pre-trained parameters to be used for prediction
            load_params (bool): Bool to signal if params should be loaded
        """
        super(PytorchAcousticBackchanneler, self).__init__()

        # First CNN
        cnn = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(11, 13), stride=(3,1))
        if load_params:
            weights = np.transpose(parameters[0][0], (3, 2, 0, 1))
            cnn.weight = torch.nn.Parameter(torch.tensor(weights).float())
            cnn.bias = torch.nn.Parameter(torch.tensor(parameters[0][1]).float())

        self.cnn1 = nn.Sequential(
            cnn,
            nn.ReLU(),
            nn.MaxPool2d((23, 1))
        )

        # Second CNN
        cnn = nn.Conv2d(in_channels=1, out_channels=16, kernel_size = (12, 13), stride=(3,1))
        if load_params:
            weights = np.transpose(parameters[1][0], (3,2,0,1))
            cnn.weight = torch.nn.Parameter(torch.tensor(weights).float())
            cnn.bias = torch.nn.Parameter(torch.tensor(parameters[1][1]).float())
        self.cnn2 = nn.Sequential(
            cnn,
            nn.ReLU(),
            nn.MaxPool2d((23, 1))
        )

        # Linear layer
        self.linear1 = nn.Linear(in_features=64, out_features=100)
        if load_params:
            self.linear1.weight = torch.nn.Parameter(torch.tensor(parameters[2][0].T).float())
            self.linear1.bias = torch.nn.Parameter(torch.tensor(parameters[2][1]).float())
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Softmax
        self.linear2 = nn.Linear(in_features=100, out_features=3)
        if load_params:
            self.linear2.weight = torch.nn.Parameter(torch.tensor(parameters[3][0].T).float())
            self.linear2.bias = torch.nn.Parameter(torch.tensor(parameters[3][1]).float())
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat_inputs):
        """
        PyTorch forward method used for training and prediction. It defines the interaction between layers.
        Args:
            feat_inputs (numpy array): It contains the network's input.

        Returns:
            out (torch.tensor): Network's output
        """
        feat_inputs = torch.tensor(feat_inputs).float()
        feat_inputs = feat_inputs.unsqueeze(1)
        cnn_1 = self.cnn1(feat_inputs)
        cnn_1 = cnn_1.flatten(1)
        cnn_2 = self.cnn2(feat_inputs).flatten(1)
        out = torch.cat((cnn_1, cnn_2), 1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out
