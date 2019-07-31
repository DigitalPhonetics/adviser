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

"""This module provides noise modules mainly for training and evaluating policies."""

from abc import abstractmethod
from typing import List

from dialogsystem import DialogSystem
from modules.module import Module
from utils import UserAct, UserActionType, common
from utils.logger import DiasysLogger
from utils.domain.jsonlookupdomain import JSONLookupDomain


class Noise(Module):
    """The base class for every noise module implementing a forward function.

    This class takes care of permuting an intent, slot or value depending on a probability.
    The functions permute_* need to be implemented by the inheriting class.

    Args:
        domain (Domain): The domain for which the user simulator will be instantiated.
            It will probably only work within this domain.
        train_error_rate (float): The training error rate. 1 equals 100% train error.
        test_error_rate (float): The testing error rate. 1 equals 100% testing error.
        pdf (List[float]): The pdf for either permuting value, slot or intent (in this order).

    """
    # probabilities are in the following order: permute_value, permute_slot, permute_intent
    # pylint: disable=dangerous-default-value
    def __init__(self, domain: JSONLookupDomain, train_error_rate: float, test_error_rate: float,
                 pdf: List[float] = [0.7, 0.3, 0.0], logger: DiasysLogger = DiasysLogger()):
        super(Noise, self).__init__(domain, logger=logger)
        self.domain = domain
        self.train_error_rate = train_error_rate
        self.test_error_rate = test_error_rate
        self.backup = train_error_rate, test_error_rate
        self.pdf = pdf
        self.pdf_wo_value = [float(i)/sum(self.pdf[1:]) for i in self.pdf[1:]]

    # pylint: disable=arguments-differ
    def forward(self, dialog_graph: DialogSystem, user_acts: List[UserAct], **kwargs)\
            -> dict(user_acts=List[UserAct]):
        """
        Permutes the given user actions, therefore 1) it is determined whether a user action will
        be permuted at all (depending on training/testing) and 2) chooses the value, slot or intent
        from a pdf for permutation.

        Args:
            dialog_graph (DialogSystem): The calling instance of the dialog system; used to
                determine the current turn.
            user_acts (List[UserAct]): The system action for which a user response will be
                retrieved.
            kwargs (dict): Any other arguments in the pipeline.

        Returns:
            Dict[user_acts=List[UserAct]]: Dictionary including the noisy/permuted user actions as
                a list.

        """
        # degrade user action (if applicable)
        change = False
        for action in user_acts:
            if (self.is_training and common.random.random() < self.train_error_rate or
                    not self.is_training and common.random.random() < self.test_error_rate):
                change = True # indicate noise

                if action.slot is None:
                    choice = -1 # TODO should be 2, but intent permutation is not fully implemented currently
                    change = False # remove once TODO is finished
                elif action.value is None:
                    choice = common.numpy.random.choice([0, 1], p=self.pdf_wo_value) # TODO should be [1,2], but intent permutation is not fully implemented currently
                else:
                    choice = common.numpy.random.choice([0, 1, 2], p=self.pdf)

                if choice == 0:
                    self.permute_value(action)
                elif choice == 1:
                    self.permute_slot(action)
                elif choice == 2:
                    self.permute_intent(action)

        if change:
            self.logger.dialog_turn(f"Noisy User Actions {user_acts}")
            
        return {'user_acts': user_acts}

    def update(self, train_error_rate: float = None, test_error_rate: float = None):
        """
        Changes the current training or testing error rate.

        Args:
            train_error_rate (float): The error rate while training..
            test_error_rate (float): The error rate while testing.

        """
        if train_error_rate is not None:
            self.train_error_rate = train_error_rate
        if test_error_rate is not None:
            self.test_error_rate = test_error_rate

    def deactivate(self):
        """Disables any noise from this module."""
        self.backup = (self.train_error_rate, self.test_error_rate)
        self.update(0.0, 0.0)

    def reset(self):
        """
        Resets the noise probability to whatever it was before deactivation or
        (if it wasn't deactivated) to the value on initialisation.
        """
        self.train_error_rate, self.test_error_rate = self.backup

    @abstractmethod
    def permute_intent(self, action):
        """Abstract method, needs to be overridden by subclass."""
        raise NotImplementedError

    @abstractmethod
    def permute_slot(self, action):
        """Abstract method, needs to be overridden by subclass."""
        raise NotImplementedError

    @abstractmethod
    def permute_value(self, action):
        """Abstract method, needs to be overridden by subclass."""
        raise NotImplementedError

class SimpleNoise(Noise):
    """Simple noise which permutes intent, slot and value by drawing uniformly from the
    possibilities.

    Args:
        domain (Domain): The domain for which the user simulator will be instantiated.
            It will probably only work within this domain.
        train_error_rate (float): The training error rate. 1 equals 100% train error.
        test_error_rate (float): The testing error rate. 1 equals 100% testing error.
        pdf (List[float]): The pdf for either permuting value, slot or intent (in this order).

    """
    def permute_intent(self, action):
        """**CURRENTLY DEACTIVATED**

        Permutes the intent of an action by chossing from all other intents uniformly. If the new
        intent needs a slot or value, appropriate values are drawn uniformly.

        Args:
            action (UserAct): The action for which the intent will be permuted.

        """
        return
        candidates = list(UserActionType)

        # exclude some actions (some are legacy)
        if UserActionType.Confirm in candidates:
            candidates.remove(UserActionType.Confirm)
        if UserActionType.Deny in candidates:
            candidates.remove(UserActionType.Deny)
        if UserActionType.Affirm in candidates:
            candidates.remove(UserActionType.Affirm)

        probs = common.numpy.ones(len(candidates))
        probs[candidates.index(action.type)] = 0 # exclude current intent
        probs = probs / sum(probs)
        # choose and set new intent
        action.type = common.numpy.random.choice(candidates, p=probs)
        # TODO make sure to choose new slots and values (if applicable)

    def permute_slot(self, action):
        """
        Permutes the intent of an action by chossing from all other intents uniformly. If the new
        intent needs a slot or value, appropriate values are drawn uniformly.

        Args:
            action (UserAct): The action for which the slot will be permuted.

        """
        if action.slot is not None: # TODO what if None?
            if action.type == UserActionType.Request:
                candidates = self.domain.get_requestable_slots()[:]
            else:
                candidates = list(self.domain.get_informable_slots())[:]

            candidates.remove(action.slot) # exclude current slot
            action.slot = common.random.choice(candidates) # choose new slot

            if action.type in [UserActionType.Inform, UserActionType.NegativeInform]:
                # NOTE maybe let bst check for consistency?
                # choose new value for new slot TODO randomly?
                action.value = common.random.choice(self.domain.get_possible_values(action.slot))

    def permute_value(self, action):
        """
        Permutes the intent of an action by chossing from all other intents uniformly. If the new
        intent needs a slot or value, appropriate values are drawn uniformly.

        Args:
            action (UserAct): The action for which the value will be permuted.

        """
        if action.value is not None:
            candidates = self.domain.get_possible_values(action.slot)[:]
            candidates.append('dontcare') # account for 'dontcare' slot
            candidates.remove(action.value) # exclude current value
            action.value = common.random.choice(candidates) # choose new value
