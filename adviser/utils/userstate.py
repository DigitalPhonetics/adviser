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

""" This module provides the UserState class. """

import copy
from enum import Enum


class EngagementType(Enum):
    """The type for a user engagement as used in :class:`UserState`."""

    High = "high"
    Low = "low"


class EmotionType(Enum):
    """The type for a user emotion as used in :class:`UserState`."""
    Happy = "happy"
    Sad = "sad"
    Angry = "angry"
    Neutral = "neutral"


class UserState:
    """
    The representation of a user state.
    Can be accessed like a dictionary
    """
    def __init__(self):
        self._history = [self._init_userstate()]

    def __getitem__(self, val):  # for indexing
        # if used with numbers: int (e.g. state[-2]) or slice (e.g. state[3:6])
        if isinstance(val, int) or isinstance(val, slice):
            return self._history[val]  # interpret the number as turn
        # if used with strings (e.g. state['beliefs'])
        elif isinstance(val, str):
            # take the current turn's belief state
            return self._history[-1][val]

    def __iter__(self):
        return iter(self._history[-1])

    def __setitem__(self, key, val):
        self._history[-1][key] = val

    def __len__(self):
        return len(self._history)

    def __contains__(self, val):  # assume
        return val in self._history[-1]

    def __repr__(self):
        return str(self._history[-1])

    def start_new_turn(self):
        """
        ONLY to be called by the user state tracker at the begin of each turn,
        to ensure the correct history can be accessed correctly by other modules
        """

        # copy last turn's dict
        self._history.append(copy.deepcopy(self._history[-1]))

    def _init_userstate(self):
        """Initializes the user state based on the currently active domain

        Returns:
            (dict): dictionary of user emotion and engagement representations

        """

        # TODO: revist when we include probabilites, sets should become dictionaries
        user_state = {"engagement": EngagementType.Low,
                      "emotion": EmotionType.Neutral}

        return user_state
