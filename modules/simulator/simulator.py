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

"""This module provides user simulators mainly for training and evaluating policies."""

from typing import List

from utils.domain.domain import Domain
from dialogsystem import DialogSystem
from modules.module import Module
from utils import UserAct, SysAct, Goal
from utils.logger import DiasysLogger


class UserSimulator(Module):
    """The base class for a user simulator.

    This class provides a reset and a forward function for user simulators. Any user simulator
    should inherit from this class.

    Args:
        domain (Domain): The domain for which the user simulator will be instantiated. It will
        probably only work within this domain.

    """
    def __init__(self, domain: Domain, logger: DiasysLogger = DiasysLogger()):
        super(UserSimulator, self).__init__(domain, logger=logger)

        self.domain = domain
        self.goal = None
        self.turn = None

    def receive(self, sys_act):
        """
        Processes the received system action.
        Needs to be implemented by the inheriting class.
        """
        raise NotImplementedError

    def respond(self):
        """Returns the next user action. Needs to be implemented by the inheriting class."""
        raise NotImplementedError

    # pylint: disable=arguments-differ
    def forward(self, dialog_graph: DialogSystem, sys_act: SysAct = None, **kwargs)\
            -> dict(user_acts=List[UserAct], sim_goal=Goal):
        """
        Will use the inheriting user simulator to retrieve a user response based on the given
        system action.

        Args:
            dialog_graph (DialogSystem): The calling instance of the dialog system; used to
                determine the current turn.
            sys_act (SysAct): The system action for which a user response will be retrieved.
            kwargs (dict): Any other arguments in the pipeline.

        Returns:
            dict: Dictionary including the user acts as a list and the current user's goal.

        """
        self.turn = dialog_graph.num_turns

        if sys_act is not None:
            self.receive(sys_act)

        user_acts = self.respond()

        # user_acts = [UserAct(text="Hi!", act_type=UserActionType.Hello, score=1.)]

        self.logger.dialog_turn("User Action: " + str(user_acts))
        # input()
        return {'user_acts': user_acts, 'sim_goal': self.goal}
