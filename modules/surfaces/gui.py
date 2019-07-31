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

"""The GUI module provides ADVISER modules that can access the GUI."""

from modules.surfaces.dialoggui.IMSChatter import DialogThread, ChatInfo
from modules.module import Module
from utils.domain.domain import Domain
from utils.sysact import SysAct, SysActionType
from utils import DiasysLogger
from dialogsystem import DialogSystem


_GUI = None  # GUI singleton
def init_gui():
    """Initialises the GUI application.

    If the GUI has already been initialised, nothing is done.
    """
    global _GUI
    if _GUI is None:
        _GUI = DialogThread()
        _GUI.start()


def close_permanently():
    """Sends the GUI the message to close itself."""
    _GUI.provide(ChatInfo.Close, True)


class GuiInput(Module):
    """Gets the user utterance from the GUI.

    Waits for the GUI to return the user's message.

    Attributes:
        initial_turn: whether or not this is the first time that the forward function is called
    """

    def __init__(self, domain=None, subgraph: dict = None, 
                 logger : DiasysLogger =  DiasysLogger()):
        Module.__init__(self, domain, subgraph, logger = logger)
        self.initial_turn = True
        init_gui()  # initialise the GUI

    def forward(self, dialog_graph: DialogSystem, **kwargs) -> dict(user_utterance=str):
        """Forward function inherited from Module interface.

        If this function has not been called before, do not pass a message.
        Otherwise, it blocks the application until the user has entered a
        message in the GUI.

        Arguments:
            dialog_graph (DialogSystem): The dialog system this module is part of.
                                         Useful to access other modules,
                                         turn and dialog counters etc.
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            dict: a dict containing the user utterance which is automatically added to the kwargs
        """
        if self.initial_turn:
            self.initial_turn = False
            return {}

        message = _GUI.wait_for(ChatInfo.UserUtterance)
        self.logger.dialog_turn('User Utterance: %s' % message)
        return {'user_utterance': message}

    def start_dialog(self, **kwargs):
        """Restarts the GUI.

        This method is used in case of multiple dialogs, e.g. for evaluation.
        It simply clears the dialog GUI widget and resets initial_turn to True.

        Args:
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            dict: an empty dict
        """
        _GUI.provide(ChatInfo.Clear, True)
        self.initial_turn = True
        return {}


class GuiOutput(Module):
    """Writes the system utterance to the console."""

    def __init__(self, domain: Domain = None, subgraph: dict = None, 
                 logger : DiasysLogger =  DiasysLogger()):
        Module.__init__(self, domain, subgraph, logger = logger)
        init_gui()

    def forward(self, dialog_graph: DialogSystem, sys_utterance: str = '', **kwargs) -> dict():
        """Forward function inherited from Module interface.

        The message is simply added to the graphical user interface.

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
        self.logger.dialog_turn('System Utterance: %s' % sys_utterance)
        _GUI.provide(ChatInfo.SystemUtterance, sys_utterance)
        return {}


class Feedback(Module):
    """Displays the feedback inside the GUI (for user evaluation)

    The module blocks while the user completes the form.
    It will automatically appear at the end of a dialogue.

    Attributes:
        dialogs (int): dialogue counter
    """

    def __init__(self, domain: Domain = None, subgraph: dict = None, 
                 logger : DiasysLogger =  DiasysLogger()):
        Module.__init__(self, domain, subgraph, logger = logger)
        init_gui()
        self.dialogs = 0

    def forward(self, dialog_graph: DialogSystem, sys_act: SysAct = None, **kwargs) -> dict():
        """Forward function inherited from Module interface.

        If this function is called and the user just entered a 'Bye' action,
        this method will display the feedback form and block while the user
        completes the form.

        Args:
            dialog_graph (DialogSystem): The dialog system this module is part of.
                                         Useful to access other modules,
                                         turn and dialog counters etc.
            sys_act (SysAct): The system act, to check whether the dialogue was finished
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            dict: an empty dict, since nothing is added to the kwargs
        """
        # check whether the dialogue was finished
        if sys_act is not None and sys_act.type == SysActionType.Bye:
            # send message to the GUI
            _GUI.provide(ChatInfo.FeedbackRequest, 'Feedback for Task %d' % self.dialogs)
            feedback = _GUI.wait_for(ChatInfo.FeedbackSent)  # wait for the reply
            self.logger.info("""
############ Manual Evaluation Summary #################
Success:    %d/5
Quality:    %d/7
Comment:    %s

""" % feedback)

        return {}

    def start_dialog(self, **kwargs):
        """Called at the start of each dialogue, inherited from Module

        Counts the number of dialogues.

        Args:
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            dict: an empty dict, since nothing is added to the kwargs
        """
        self.dialogs += 1
        return {}


class TaskDescription(Module):
    """Displays the task description inside the GUI

    The module blocks until the user has read the description.
    It will automatically appear at the start of a dialogue.

    Attributes:
        initial_turn (bool): whether this is the first turn of the dialogue
        description (str): what should be shown to the user
        dialogs (int): dialogue counter
    """

    def __init__(self, domain: Domain = None, subgraph: dict = None, 
                 logger : DiasysLogger =  DiasysLogger()):
        Module.__init__(self, domain, subgraph, logger = logger)
        self.initial_turn = True
        self.description = ''
        init_gui()
        self.dialogs = 0

    def set_task_description(self, description):
        """Setter for the task description

        Arguments:
            description (str): the description of the task
        """
        self.description = description

    def forward(self, dialog_graph: DialogSystem, **kwargs) -> dict():
        """Forward function inherited from Module interface.

        When this function is called for the first time, this method will
        display the task description and block until the user has clicked
        the task-read button.

        Args:
            dialog_graph (DialogSystem): The dialog system this module is part of.
                                         Useful to access other modules,
                                         turn and dialog counters etc.
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            an empty dict, since nothing is added to the kwargs
        """
        if self.initial_turn:
            # send message to the GUI
            _GUI.provide(ChatInfo.TaskDescription, (self.description, 'Task %d' % self.dialogs))
            _GUI.wait_for(ChatInfo.TaskRead)  # wait for a reply
            self.initial_turn = False
        return {}

    def start_dialog(self, **kwargs):
        """Procedure to be executed right after the dialog started.

        Args:
            **kwargs (dict): Dictionary of information passing across modules,
                             keys and values are defined according to the modules'
                             definition and interaction

        Returns:
            dict: an empty dict, since nothing is added to the kwargs
        """
        self.dialogs += 1
        self.initial_turn = True

        return {}
