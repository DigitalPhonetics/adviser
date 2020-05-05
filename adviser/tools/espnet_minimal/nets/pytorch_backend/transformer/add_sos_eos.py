#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unility functions for Transformer."""

import torch


def add_sos_eos(ys_pad, sos, eos, ignore_id):
    """
    Add `<sos>` and `<eos>` labels.

    Arguments:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of `<sos>`
        eos (int): index of `<eos>`
        ignore_id (int): index of padding

    Returns:
        torch.Tensor: padded tensor (B, Lmax)
    """
    from tools.espnet_minimal import pad_list
    _sos = ys_pad.new([sos])
    _eos = ys_pad.new([eos])
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)
