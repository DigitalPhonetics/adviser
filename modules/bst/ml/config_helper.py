###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
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

import configparser
import os

import torch

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'data_path.cfg')

def set_device(use_cuda, gpu=0):
    global DEVICE
    if use_cuda == True:
        DEVICE = torch.device("cuda:" + str(gpu))
    else:
        DEVICE = torch.device("cpu")
set_device(False)   # default

def get_device():
    return DEVICE

def get_data_path():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config['DEFAULT']['data_path']


def get_preprocessing_path():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config['DEFAULT']['preprocessing_path']


def get_runs_path():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config['DEFAULT']['runs_path']


def get_weight_path():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config['DEFAULT']['weigths_path']