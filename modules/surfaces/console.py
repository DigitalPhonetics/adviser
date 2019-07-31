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

"""The console module provides ADVISER modules that access the console for input and output."""

from modules.module import Module
from utils.domain.domain import Domain
from dialogsystem import DialogSystem
from utils import DiasysLogger
from utils.common import Language


class ConsoleInput(Module):
    """Gets the user utterance from the console.

    Waits for the built-in input function to return a non-empty text.

    Attributes:
        initial_turn (bool): whether or not this is the first time that
                             the forward function is called
    """

    def __init__(self, domain: Domain = None, subgraph: dict = None,
                 logger: DiasysLogger = DiasysLogger(), language: Language = None):
        Module.__init__(self, domain, subgraph, logger=logger)  # independent of the domain
        self.language = language

    def forward(self, dialog_graph: DialogSystem, **kwargs) -> dict(user_utterance=str):
        """Forward function inherited from Module interface.

        If this function has not been called before, do not pass a message.
        Otherwise, it blocks the application until the user has entered a
        valid (i.e. non-empty) message in the console.

        Args:
            dialog_graph (DialogSystem): The dialog system this module is part of.
                                         Useful to access other modules,
                                         turn and dialog counters etc.
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            dict: a dict containing the user utterance which is automatically added
                  to the kwargs
        """

        utterance = ''
        if self.language == None:
            self.language = self._set_language()
        else:
            if dialog_graph.num_turns > 0:
                while utterance.strip() == '':
                    utterance = input('>>> ')  # this method blocks
                self.logger.dialog_turn('User Utterance: %s' % utterance)
        return {'user_utterance': utterance, 'language': self.language}

    def _set_language(self) -> Language:
        """
            asks the user to select the language of the system, returning the enum
            representing their preference, or None if they don't give a recognized
            input

        """

        utterance = ""
        print("Please select your language: English or German")
        while utterance.strip() == "":
            utterance = input(">>> ")
        utterance = utterance.lower()
        if utterance == 'e' or utterance == 'english':
            return Language.ENGLISH
        elif utterance == 'g' or utterance == 'german' or utterance == 'deutsch'\
                or utterance == 'd':
            return Language.GERMAN
        else:
            return None


class ConsoleOutput(Module):
    """Writes the system utterance to the console."""

    def __init__(self, domain: Domain = None, subgraph: dict = None,
                 logger: DiasysLogger = DiasysLogger()):
        Module.__init__(self, domain, subgraph, logger=logger)

    def forward(self, dialog_graph: DialogSystem, sys_utterance: str = None, sys_act: str = None, **kwargs) -> dict():
        """Forward function inherited from Module interface.

        The message is simply printed to the console.

        Args:
            dialog_graph (DialogSystem): The dialog system this module is part of.
                                         Useful to access other modules, 
                                         turn and dialog counters etc.
            sys_utterance (str): The system utterance, as added to the kwargs by the NLG module
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            dict: an empty dict, since nothing is added to the kwargs

        Raises:
            ValueError: if there is no system utterance to print
        """
        if sys_utterance is not None and sys_utterance != "":
            print("System: {}".format(sys_utterance))
        elif sys_act is not None:
            print("System utterance missing, falling back to system action on intention level.")
            print("System: {}".format(sys_act))
        else:
            raise ValueError("There is no system utterance. Did you forget to call an NLG module before?")
        # self.logger.dialog_turn('System Utterance: %s' % sys_utterance)
        return {}
