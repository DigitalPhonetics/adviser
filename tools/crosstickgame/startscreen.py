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

import sys
import os

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal
from tools.crosstickgame.editwidget import EditWidget
from tools.crosstickgame.persistence.taskcreator import TaskCreator


class StartScreen(QMainWindow):
    """The main window of this application

    At start, it only consists of an empty QTabWidget.
    Tasks can the be added via the task creation dialog or
    the open dialog.
    
    Attributes:
        _creation_dialog: the task creation dialog
    """

    def __init__(self):
        QMainWindow.__init__(self)
        self._ui = uic.loadUi(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui/startscreen.ui'), self)
        
        self._ui.new_task.triggered.connect(self._on_new_task)
        self._ui.save.triggered.connect(self._save)
        self._ui.load.triggered.connect(self._load)

        self._creation_dialog = TaskCreator()
        self._creation_dialog.accepted.connect(self._add_task)

        self._ui.tasks.tabCloseRequested.connect(self._on_close_tab)

        # implement Ctrl+S
        QShortcut(QKeySequence.Save, self).activated.connect(self._save)

    def _on_close_tab(self, idx):
        """Callback function for when a user clicks the "X" of a tab.

        The tab is then simply removed from the QTabWidget.
        
        Arguments:
            idx {int} -- The index of the task within the QTabWidget
        """
        self._ui.tasks.removeTab(idx)

    def _on_new_task(self):
        """Callback function when "New Task..." is clicked by the user.
        """
        self._creation_dialog.open()

    def _add_task(self):
        """Callback function when a task has been created by the user.
        """
        task, name = self._creation_dialog.get_last_task()
        self._ui.tasks.addTab(task, name)
    
    def _load(self):
        """'Open...' implementation for the MinMax game.

        The user can select a JSON file that has been created in the past using
        the task creation dialog.
        """
        dest_file, _ = QFileDialog().getOpenFileName(self, 'Select a JSON-file', filter='JSON-Files (*.json)')
        if dest_file == '':
            return
        self._ui.tasks.addTab(EditWidget.load(dest_file), os.path.basename(dest_file))
    
    def _save(self):
        # each task should have its own implemented save function
        self._ui.tasks.currentWidget().save()
