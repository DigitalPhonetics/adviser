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

import os

from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QDialogButtonBox


class TaskCreator(QDialog):
    """Dialog which shows the user possible tasks and let them choose one of them
    
    This class consists of a tab widget in which all possible tasks are listed.
    Currently, this is unnecessary since there is only one "task" - the MinMaxGame.
    However, since we probably want to have more visual tools in the future,
    I decided to make the approach easily extendable for other tasks.
    """

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self._ui = uic.loadUi(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../ui/taskcreator.ui'), self)

        # all possible tasks have to know about the callback function
        # to decide whether the current configuration is valid
        self._ui.testcases.set_callback(self._on_task_changed)
        self._ui.regexes.set_callback(self._on_task_changed)
        
        self._ui.tab_widget.currentChanged.connect(self._on_tab_changed)

    def open(self):
        """Method to show the task selection to the user
        """
        QDialog.open(self)
        # at the beginning, the callback function has to be called manually
        self._on_task_changed(self._ui.tab_widget.currentWidget())

    def _on_tab_changed(self, _):
        """Callback function for when the current task is changed

        Currently, we only have to update the OK button.
        """
        self._on_task_changed(self._ui.tab_widget.currentWidget())

    def _on_task_changed(self, widget):
        """Callback function for when something inside the task configuration changes.

        This method decides whether or not the OK button is enabled by calling
        the is_valid method of the task. Therefore, this callback should always
        be called when anything inside the form has been changed. 
        
        Arguments:
            widget {QWidget} -- the widget that changed
        """
        if self._ui.tab_widget.currentWidget() == widget:
            self._ui.buttons.button(QDialogButtonBox.Ok).setEnabled(widget.is_valid())

    def get_last_task(self):
        """Creates the widget of the task.

        This method should be called immediately after the OK button has been
        pressed. The returned widget should be added to the QTabWidget of the
        start screen.
        
        Returns:
            QWidget -- the widget of the task that has just been created
        """
        return self._ui.tab_widget.currentWidget().create_task()
