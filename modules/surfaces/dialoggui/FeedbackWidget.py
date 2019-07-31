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


class FeedbackWidget(QWidget):
    """
    Widget showing the user the feedback form for evaluation.
    It emits the feedback_sent signal when the user has created a valid feedback
    and clicked the send button.
    """
    feedback_sent = pyqtSignal(int, int, str)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.__ui = uic.loadUi(os.path.join(UI_DIR, 'feedback.ui'), self)

        self.__ui.send.clicked.connect(self._send_feedback)
        self.__ui.success_slider.valueChanged.connect(self._on_success_changed)
        self.__ui.quality_slider.valueChanged.connect(self._on_quality_changed)
    
    def set_header(self, text):
        """
        :param text: the title of the feedback form, e.g. 'Feedback for task 3/5'
        """
        self.__ui.header.setText(text)
    
    def _decide_send_enabled(self):
        """
        Method for deciding whether the feedback is valid, i.e. the user has selected
        values for success (1-5) and quality (1-7).
        """
        self.__ui.send.setEnabled(self.__ui.success_slider.value() > 0 and
                                  self.__ui.quality_slider.value() > 0)
    
    def _on_success_changed(self, value):
        """
        Callback function when the user dragged the success slider to a new value
        
        :param value: value to which the user has dragged the slider
        """
        self.__ui.success_display.setText('%d/5' % value)
        self._decide_send_enabled()  # update the send button
    
    def _on_quality_changed(self, value):
        """
        Callback function when the user dragged the quality slider to a new value
        
        :param value: value to which the user has dragged the slider
        """
        self.__ui.quality_display.setText('%d/7' % value)
        self._decide_send_enabled()
    
    def _send_feedback(self):
        """
        Callback function when the user has clicked the (enabled) submit button
        """
        # read feedback
        success = self.__ui.success_slider.value()
        quality = self.__ui.quality_slider.value()
        comments = self.__ui.comments.toPlainText()

        # reset the form
        self.__ui.success_slider.setValue(0)
        self.__ui.quality_slider.setValue(0)
        self.__ui.comments.setPlainText('')
        self._decide_send_enabled()

        self.feedback_sent.emit(success, quality, comments)  # inform application
