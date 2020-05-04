#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
# matplotlib related
import os

# io related
import matplotlib
import torch

matplotlib.use('Agg')


def add_gradient_noise(model, iteration, duration=100, eta=1.0, scale_factor=0.55):
    """Adds noise from a standard normal distribution to the gradients.

    The standard deviation (`sigma`) is controlled by the three hyper-parameters below.
    `sigma` goes to zero (no noise) with more iterations.

    Args:
        model (torch.nn.model): Model.
        iteration (int): Number of iterations.
        duration (int) {100, 1000}: Number of durations to control the interval of the `sigma` change.
        eta (float) {0.01, 0.3, 1.0}: The magnitude of `sigma`.
        scale_factor (float) {0.55}: The scale of `sigma`.
    """
    interval = (iteration // duration) + 1
    sigma = eta / interval ** scale_factor
    for param in model.parameters():
        if param.grad is not None:
            _shape = param.grad.size()
            noise = sigma * torch.randn(_shape).to(param.device)
            param.grad += noise


# * -------------------- general -------------------- *
def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json).

    Args:
        model_path (str): Model path.
        conf_path (str): Optional model config path.

    Returns:
        list[int, int, dict[str, Any]]: Config information loaded from json file.

    """
    if conf_path is None:
        model_conf = os.path.dirname(model_path) + '/model.json'
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info('reading a config file from ' + model_conf)
        confs = json.load(f)
    if isinstance(confs, dict):
        # for lm
        args = confs
        return argparse.Namespace(**args)
    else:
        # for asr, tts, mt
        idim, odim, args = confs
        return idim, odim, argparse.Namespace(**args)


def torch_save(path, model):
    """Save torch model states.

    Args:
        path (str): Model path to be saved.
        model (torch.nn.Module): Torch model.

    """
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def torch_load(path, model):
    """Load torch model states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.

    """
    if 'snapshot' in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    # debugging:
    # print(model_state_dict)

    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict
