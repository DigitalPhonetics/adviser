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

from utils import UserActionType
from utils.sysact import SysActionType
from utils.logger import DiasysLogger


class DialogSystem(object):
    """This is the main dialog system, holding all modules and taking care of
    the data flow.

    Public methods:
    train -- trains the dialog system
    chat -- allows to chat with the dialog system
    eval -- Evaluates the dialog system

    Instance variables:
    current_domain -- the currently active domain
    modules -- a list of modules which will be called in the given order
    sequentially
    """

    def __init__(self, *modules, domain=None, logger: DiasysLogger = DiasysLogger()):
        self.domain = domain
        self.logger = logger
        self.modules = modules

        self.is_training = False
        self.num_dialogs = 0
        self.num_turns = 0

    def _start_dialog(self, kwargs: dict = None):
        kwargs = kwargs or {}
        self.logger.dialog_turn("# DIALOG {} STARTED #".format(self.num_dialogs))
        for module in self.modules:
            # kwargs = module.start_dialog()
            kwargs = {**kwargs, **module.start_dialog(**kwargs)}
        return kwargs

    def _end_dialog(self, kwargs: dict):
        # TODO find a better way; maybe evaluate outside in extra module?
        if 'sim_goal' in kwargs:
            sim_goal = kwargs['sim_goal']
        else:
            sim_goal = None

        # end of dialog
        for module in self.modules:
            module.end_dialog(sim_goal)

    def run_dialog(self, max_length=-1):
        """ Perform one complete dialog.

        Args:
            max_length (int): end dialog after the specified amount of turns if max_length > 0
        """

        # start of dialog
        kwargs = self._start_dialog()

        self.num_turns = 0
        # try:
        while True:
            if self.num_turns == max_length:
                self.logger.dialog_turn("Maximum dialog length reached, ending dialog.")
                break

            kwargs, stop = self._forward_turn(kwargs)
            if stop:
                break

        # end of dialog
        self._end_dialog(kwargs)

        self.logger.dialog_turn("# DIALOG {} FINISHED #".format(self.num_dialogs))
        self.num_dialogs += 1
        # except:
        #     logger.error("Fatal error in main loop", exc_info=True)

    def _forward_turn(self, kwargs):
        """ Forward one turn of a dialog. """
        self.logger.dialog_turn("# TURN " + str(self.num_turns) + " #")

        # call each module in the list
        for module in self.modules:
            kwargs = {**kwargs, **module.forward(self, **kwargs)}

        stop = False
        if 'user_acts' in kwargs and kwargs['user_acts'] is not None:
            for action in kwargs['user_acts']:
                if action.type is UserActionType.Bye:
                    stop = True
            if 'sys_act' in kwargs and kwargs['sys_act'].type == SysActionType.Bye:
                stop = True

        self.num_turns += 1

        return kwargs, stop

    def train(self):
        """ Configure all modules in the dialog graph for training mode. """
        self.is_training = True

        # set train flag in each module
        for module in self.modules:
            module.train()

    def eval(self):
        """ Configure all modules in the dialog graph for evaluation mode. """
        self.is_training = False

        # set train flag in each module
        for module in self.modules:
            module.eval()
