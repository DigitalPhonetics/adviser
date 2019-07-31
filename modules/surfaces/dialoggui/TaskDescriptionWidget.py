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
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal
from modules.surfaces.dialoggui.IMSChatter import UI_DIR


class TaskDescriptionWidget(QWidget):
    """
    A widget that shows the description of a task for evaluation purposes

    The widget consists of a header label (e.g. for task 3/5)
    and a label containing the task description
    """
    task_read = pyqtSignal()

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.__ui = uic.loadUi(os.path.join(UI_DIR, 'TaskDescription.ui'), self)

        self.__ui.send.clicked.connect(self.task_read.emit)
    
    def set_description(self, text):
        self.__ui.description.setText('%s\n%s' % (text, 'To end the dialog, type "Bye".'))
    
    def set_header(self, text):
        self.__ui.header.setText(text)
