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

"""The console module provides ADVISER modules that access the console for input and output."""
import math
import os
import sys
import time

from services.service import PublishSubscribe
from services.service import Service
from utils.common import Language
from utils.domain import Domain
from utils.topics import Topic


class ConsoleInput(Service):
    """
    Gets the user utterance from the console.

    Waits for the built-in input function to return a non-empty text.
    """

    def __init__(self, domain: Domain = None, conversation_log_dir: str = None, language: Language = None):
        Service.__init__(self, domain=domain)
        # self.language = language
        self.language = Language.ENGLISH
        self.conversation_log_dir = conversation_log_dir
        self.interaction_count = 0
        # if self.language is None:
        #     self.language = self._set_language()

    def dialog_start(self):
        self.interaction_count = 0

    @PublishSubscribe(sub_topics=[Topic.DIALOG_END], pub_topics=["gen_user_utterance"])
    def get_user_input(self, dialog_end: bool = True) -> dict(user_utterance=str):
        """

        If this function has not been called before, do not pass a message.
        Otherwise, it blocks the application until the user has entered a
        valid (i.e. non-empty) message in the console.

        Returns:
            dict: a dict containing the user utterance
        """
        if dialog_end:
            return

        utterance = self._input()
        # write into logging directory
        if self.conversation_log_dir is not None:
            with open(os.path.join(self.conversation_log_dir, (str(math.floor(time.time())) + "_user.txt")),
                      "w") as convo_log:
                convo_log.write(utterance)
        return {'gen_user_utterance': utterance}

    def _input(self):
        "Helper function for reading text input from the console"
        utterance = ''
        try:
            sys.stdout.write('>>> ')
            sys.stdout.flush()
            line = sys.stdin.readline()
            while line.strip() == '' and not getattr(self, '_dialog_system_parent').terminating():
                line = sys.stdin.readline()
            utterance = line
            if getattr(self, '_dialog_system_parent').terminating():
                sys.stdin.close()
            return utterance
        except:
            return utterance

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
        elif utterance == 'g' or utterance == 'german' or utterance == 'deutsch' \
                or utterance == 'd':
            return Language.GERMAN
        else:
            return None


class ConsoleOutput(Service):
    """Writes the system utterance to the console."""

    def __init__(self, domain: Domain = None):
        Service.__init__(self, domain=domain)

    @PublishSubscribe(sub_topics=["sys_utterance"], pub_topics=[Topic.DIALOG_END])
    def print_sys_utterance(self, sys_utterance: str = None) -> dict():
        """

        The message is simply printed to the console.

        Args:
            sys_utterance (str): The system utterance

        Returns:
            dict with entry dialog_end: True or False

        Raises:
            ValueError: if there is no system utterance to print
        """
        if sys_utterance is not None and sys_utterance != "":
            print("System: {}".format(sys_utterance))
        else:
            raise ValueError("There is no system utterance. Did you forget to call an NLG module before?")

        return {Topic.DIALOG_END: 'bye' in sys_utterance}
