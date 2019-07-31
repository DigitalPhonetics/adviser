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
from enum import Enum
import threading
import queue

from PyQt5 import uic
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QLabel, QSpacerItem, QSizePolicy, QWidget
from PyQt5.QtCore import QTimer, QSize, pyqtSignal, pyqtSlot, QThread


class ChatInfo(Enum):
    """
    Enum for (a)synchronous communication with the modules
    """
    UserUtterance = 'user_utt'
    SystemUtterance = 'sys_utt'
    Clear = 'clear'
    Close = 'close'
    FeedbackSent = 'feedback_sent'
    FeedbackRequest = 'request_feedback'
    TaskRead = 'task_sent'
    TaskDescription = 'request_task'

UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'res/ui/'))

IN_QUEUE = queue.Queue()
OUT_QUEUE = queue.Queue()

class IMSChatter(QMainWindow):
    """
    Main widget for the chat application with an input field for the user to chat with a dialog system.
    The design is supposed to be similar to well-known chat apps like WhatsApp, Telegram, ICQ etc.

    Even though it is usually desired to start a Qt app as the main thread of an application,
    we do not use it here due to code design consistency.
    Instead, the Qt application and ADvISER's GUI modules communicate via Queue objects.
    The IN_QUEUE perceives values (including commands, i.e. ChatInfo enum values)
    from the modules.
    This app writes commands and values to the OUT_QUEUE that can be perceived by the modules,
    e.g. the user utterances.
    """

    def __init__(self):
        """
        Constructor, initialising the Qt data structure and loading the Qt Designer file.
        """
        QMainWindow.__init__(self)
        self.__ui = uic.loadUi(os.path.join(UI_DIR, 'IMSChatter.ui'), self)

        self.__ui.feedback.feedback_sent.connect(self._on_feedback_sent)
        self.__ui.task.task_read.connect(self._on_task_read)
        self.__ui.widgets.setCurrentIndex(1)  # default is the "normal" dialogue widget
        
        # listen to messages from the ADvISER modules
        self.__queue_thread = QueueThread(IN_QUEUE)
        self.__queue_thread.new_value.connect(self.handle_queue_value)
        self.__queue_thread.start()
    
    def handle_queue_value(self, command, value):
        """
        This method is called when a message has been received from the modules.
        """
        if command == ChatInfo.SystemUtterance:  # standard case
            self.__ui.chat.add_message(value)
        elif command == ChatInfo.Clear:  # for multiple dialogues
            self.__ui.chat.clear()
        elif command == ChatInfo.Close:  # when the last dialogue ended
            self.__queue_thread.requestInterruption()
            IN_QUEUE.put((ChatInfo.Close, True))  # to close the potentially blocking get
            self.close()
        elif command == ChatInfo.FeedbackRequest:  # for evaluation
            self.__ui.feedback.set_header(value)
            self.__ui.widgets.setCurrentIndex(2)  # change the widget
        elif command == ChatInfo.TaskDescription:  # for evaluation
            # value is a pair of strings for header and description text
            self.__ui.task.set_description(value[0])
            self.__ui.task.set_header(value[1])
            self.__ui.chat.set_task_description(value[0], show_task=True)
            self.__ui.widgets.setCurrentIndex(0)  # change the widget
    
    def _on_feedback_sent(self, success, quality, comments):
        """
        Callback function when the user clicked the send button in the feedback form

        :param success: success of the dialogue on a 1 to 5 scale
        :param quality: quality of the dialogue on a 1 to 7 scale
        :param comments: additional comments of the user
        """
        # send feedback to the modules
        OUT_QUEUE.put((ChatInfo.FeedbackSent, (success, quality, comments)))
        self.__ui.widgets.setCurrentIndex(1)  # back to "normal" dialogue widget
    
    def _on_task_read(self):
        """
        Callback function when the user clicked the send button in the task description page
        """
        # inform the modules that the task description page has been sent
        OUT_QUEUE.put((ChatInfo.TaskRead, True))
        self.__ui.widgets.setCurrentIndex(1)  # back to "normal" dialogue widget
    

class DialogThread(threading.Thread):
    """
    A new thread outside of main ADvISER thread handling the GUI application

    This thread is meant to be used by the ADvISER modules to start the GUI
    as well as to send and receive messages to/from the GUI.
    The messages can be send and received in the main ADvISER thread and will
    automatically be synchronised with the GUI application thread using queues.

    This procedure is necessary since changes of the GUI widgets may only
    take place within the GUI application thread. Putting both GUI and ADvISER
    modules together in one thread would require the GUI application thread
    to be the main thread. This would violate ADvISER's modular structure.
    Also, ADvISER does not require a GUI, so the application thread should
    only be started on demand.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        # stack for incoming (from GUI to modules) messages in case that the
        # next incoming message is not the one being waited for
        self.incoming = []
    
    def provide(self, command, value):
        """
        This method can be called by the modules to send a message to the GUI.
        It can be called from the ADvISER main thread and will automatically be synchronised.

        :param command: the command name (should be an element from the ChatInfo enum)
        :param value: the value, e.g. the text of a system utterance.
        """
        IN_QUEUE.put((command, value))  # message is always a pair of command and value
    
    def wait_for(self, command):
        """
        Blocks until the given command has been received from the GUI.
        This method can be called from the ADvISER main thread and will automatically be synchronised.

        :param command: the command name (should be an element from the ChatInfo enum)
        """
        # check the stack of past incoming messages first
        for i in range(len(self.incoming)):
            if self.incoming[i][0] == command:
                return self.incoming.pop(i)[1]
        
        elem = OUT_QUEUE.get()
        while elem[0] != command:  # only add to stack if the message is not the desired command
            self.incoming.append(elem)
            elem = OUT_QUEUE.get()
        return elem[1]
    
    def run(self):
        """
        In the thread, the GUI application is started
        """
        app = QApplication(sys.argv)
        window = IMSChatter()
        window.show()
        app.exec_()


class QueueThread(QThread):
    """
    A thread within the GUI application thread that listens to incoming
    (from the modules to the GUI) messages. For each message, the
    new_value signal is emitted. Thanks to the signal/slot implementation
    of Qt, the actual changes can take place within the GUI application
    thread. The GUI must create a slot method that receives a command
    (a ChatInfo enum element) and a value as parameters.
    """
    new_value = pyqtSignal(object, object)

    def __init__(self, queue):
        QThread.__init__(self)
        self.queue = queue

    def run(self):
        # check whether requestInterruption() was called
        while not self.isInterruptionRequested():
            command, value = self.queue.get()  # blocks until new message has arrived
            self.new_value.emit(command, value)  # inform GUI about new message
