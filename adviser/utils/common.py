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

"""This modules provides a method to seed commonly used random generators."""

import random
from enum import Enum

import numpy

GLOBAL_SEED = None


class Language(Enum):
    """Set of recognized languaged"""
    ENGLISH = 0
    GERMAN = 1


def init_random(seed: int = None):
    """
    Initializes the random generators to allow seeding.

    Args:
        seed (int): The seed used for all random generators.

    """
    global GLOBAL_SEED  # pylint: disable=global-statement
    if GLOBAL_SEED is not None:
        return

    if seed is None:
        tmp_random = numpy.random.RandomState(None)
        GLOBAL_SEED = tmp_random.randint(2**32-1, dtype='uint32')
    else:
        GLOBAL_SEED = seed

    # initialize random generators
    numpy.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    try:
        # try to load torch and initialize random generator if available
        import torch
        torch.cuda.manual_seed_all(GLOBAL_SEED)  # gpu
        torch.manual_seed(GLOBAL_SEED)  # cpu
    except ImportError:
        pass

    try:
        # try to load tensorflow and initialize random generator if available
        import tensorflow
        tensorflow.random.set_random_seed(GLOBAL_SEED)
    except ImportError:
        pass

    # check whether all calls to torch.* use the same random generator (i.e. same instance)
    # works in a short test -- MS
    # print(torch.initial_seed())

    # logger.info("Seed is {:d}".format(GLOBAL_SEED))
    return GLOBAL_SEED
