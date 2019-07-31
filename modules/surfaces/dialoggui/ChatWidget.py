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
import time

from PyQt5 import uic
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QLabel, QSpacerItem, QSizePolicy, QWidget
from PyQt5.QtCore import QTimer, QSize, pyqtSignal, pyqtSlot, QThread

from modules.surfaces.dialoggui.IMSChatter import OUT_QUEUE, ChatInfo, UI_DIR


class ChatWidget(QWidget):
    """
    Main widget for the chat application with an input field for the user to chat with a dialog system.
    The design is supposed to be similar to known chat apps like WhatsApp, Telegram, ICQ etc.
    """

    def __init__(self, parent=None):
        """
        Constructor, initialising the Qt data structure and loading the Qt Designer file.
        bot is an object inheriting from the class DialogSystem which is called whenever the user writes a message.
        """
        QWidget.__init__(self, parent)
        self.__ui = uic.loadUi(os.path.join(UI_DIR, 'ChatWidget.ui'), self)

        self.__ui.send.clicked.connect(self._message_sent)
        self.__ui.typed_text.returnPressed.connect(self._message_sent)

        # scroll to the bottom if new message was sent and scroll bar is down
        self.__ui.scroll_area.verticalScrollBar().valueChanged.connect(self._track_scroll)
        self.__ui.scroll_area.verticalScrollBar().rangeChanged.connect(self._update_scroll)

        self.__ui.typed_text.setFocus()  # cursor should be in the input field per default
        self.__ui.task.setVisible(False)

        self.__history = []  # list of messages as pairs of user/non-user message and text, sent during this session
        self.__scroll_at_bottom = True  # bool variable whether the scroll bar is at the very bottom
    
    def set_task_description(self, description, show_task=True):
        """
        Optional task description label for the widget

        :param description: text to be shown to the user
        :param show_task: whether or not the description label should be visible
        """
        self.__ui.task.setVisible(show_task)
        self.__ui.task.setText('Task description:\n%s\n%s' % (description, 'To end the dialog, type "Bye".'))

    def clear(self):
        """
        Delete all messages from the screen.
        """
        for i in reversed(range(self.__ui.messages.count()-1)):
            widget = self.__ui.messages.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.__history = []

    def add_message(self, text):
        """
        Method for creating a container inside the GUI with the given text, as a message "sent" by the dialog system
        """
        # create a BotMessage object and insert it at the end of the message area inside the GUI
        self.__ui.messages.insertWidget(len(self.__history), BotMessage(text, self.__ui.message_widget))
        self.__history.append((True, text))  # add the message to the history (True stands for non-user message)

    def _message_sent(self):
        """
        Callback function when user clicks send button or presses enter key while the cursor is in the input field
        """
        text = self.__ui.typed_text.text()  # content of the input field
        if text.strip() == '':  # if no content, don't send a message
            return

        # create a SubjectMessage object and insert it at the end of the message area inside the GUI
        self.__ui.messages.insertWidget(len(self.__history), SubjectMessage(text, self.__ui.message_widget))
        self.__history.append((False, text))  # add the message to the history (False stands for user message)
        self.__ui.typed_text.clear()  # empty the input field
        OUT_QUEUE.put((ChatInfo.UserUtterance, text))

        # reply = SYSTEM_OUTPUT_QUEUE.get()
        # request the reply of the bot message, but wait a little bit and start it another thread, so other stuff can be
        # done in the meantime (e.g. displaying the user message BEFORE the dialog system has created an answer)
        # QTimer.singleShot(100, lambda: self.add_message(reply))
        # QTimer.singleShot(100, self.wait_for_output)

    def _track_scroll(self, value):
        # if the user has moved the scroll bar, check whether it is at the very bottom
        self.__scroll_at_bottom = value == self.__ui.scroll_area.verticalScrollBar().maximum()

    def _update_scroll(self, _, __):
        # when a message was added and scroll bar was at the bottom before, go to the very bottom again, so that the
        # user can see the new message
        if self.__scroll_at_bottom:
            self.__ui.scroll_area.verticalScrollBar().setSliderPosition(self.__ui.scroll_area.verticalScrollBar().maximum())


class BotMessage(QWidget):
    """
    A Qt widget representing a message of the dialog system: left-aligned label with whitish background
    """
    def __init__(self, text, parent):
        QWidget.__init__(self, parent)
        self.parent_widget = parent

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.label = PerfectLabel(text, self.parent_widget, self)
        self.label.setWordWrap(True)
        self.label.setStyleSheet('QLabel { background-color: #EEFFFFFF; border: 2px solid #CCCCCC; border-radius: 10px; padding: 8 8 5 5; }')
        self.label.setFont(QFont('Segoe UI', 12))
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.label.adjustSize()
        layout.addWidget(self.label)
        layout.addSpacerItem(QSpacerItem(50, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def sizeHint(self):
        return self.label.size()


class SubjectMessage(QWidget):
    """
    A Qt widget representing a message sent by the user: right-aligned label with blueish background
    """
    def __init__(self, text, parent=None):
        QWidget.__init__(self, parent)
        self.parent_widget = parent

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        label = PerfectLabel(text, self.parent_widget, self)
        label.setWordWrap(True)
        # #AA88FF88
        label.setStyleSheet('QLabel { background-color: #ff89cff0; border: 2px solid white; border-radius: 10px; padding: 8 8 5 5 }')
        label.setFont(QFont('Segoe UI', 12))
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        label.adjustSize()
        label.setMinimumWidth(100)
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(label)


class PerfectLabel(QLabel):  # create a label which adapts its size optimally
    def __init__(self, text, container, parent=None):
        QLabel.__init__(self, text, parent)
        self.container = container
    
    def sizeHint(self):
        size = QLabel.sizeHint(self)
        max_width = self.container.width() * 0.8
        fm = QFontMetrics(self.font())
        fm_width = fm.width(self.text()) + 30
        return QSize(min((fm_width, max_width)), size.height())

