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

""" This module provides the base class for modules as used in the dialog system. """
from utils.domain.domain import Domain
from dialogsystem import DialogSystem

from utils.logger import DiasysLogger


class Module(object):
    """
    Main base interface for all module implementations,
    being the key for the toolkit flexibility
    """
    def __init__(self, domain: Domain, subgraph: dict = None,
                 logger: DiasysLogger = DiasysLogger()):
        self.domain = domain
        self.subgraph = subgraph
        self.logger = logger
        self.is_training = False

    def forward(self, dialog_graph: DialogSystem, **kwargs):
        """
        The main functionality of the module should be implemented here.
        Through this method the module interacts with other ones.
        Therefore, the kwargs play a fundamental role by carrying
        the information exchanged.

        Args:
            dialog_graph (DialogSystem): The dialog system this module is part of.
                                         Useful to access other modules,
                                         turn and dialog counters etc.
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:

        """
        raise NotImplementedError

    def start_dialog(self, **kwargs): # pylint: disable=unused-argument
        """
        Procedure to be executed right after the dialog started.

        Args:
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            dict: to specify
        """
        return {}

    def end_dialog(self, sim_goal):
        """
        Procedure to be executed once the end dialog signal has been raised.

        Args:
            sim_goal: Goal of user (used e.g. when training a policy against a user simulator)

        Returns:

        """

    def train(self):
        """
        Sets module and its subgraph to training mode

        Returns:

        """
        self.is_training = True
        if self.subgraph is not None:
            for module_name in self.subgraph:
                self.subgraph[module_name].train()

    def eval(self):
        """
        Sets module and its subgraph to eval mode

        Returns:

        """

        self.is_training = False
        if self.subgraph is not None:
            for module_name in self.subgraph:
                self.subgraph[module_name].eval()
            