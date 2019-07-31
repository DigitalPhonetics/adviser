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

from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QTableWidgetItem, QSizePolicy, QHeaderView
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
import os

YN_FONT = QFont('Sans Serif', 20)
PM_FONT = QFont('Consolas', 12)
PM_FONT.setBold(True)


class TestCaseView(QWidget):
    """Widget that displays the test cases written by the user
    
    Attributes:
        tests_changed {pyqtSignal} -- emited when a test case was added, deleted or edited
        _module {Module} -- the NLU module used for checking the test cases
        _user_acts {frozenset[UserAct]} -- the current expected user acts from the UserActView
        _sentences {list[string]} -- the current test sentences inside this widget
    """

    tests_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._ui = uic.loadUi(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui/testcaseview.ui'), self)
        
        # tabular is a QTableWidget but just for structural purposes
        # the user should not be able to focus anything but the test sentences
        self._ui.tabular.setFocusPolicy(Qt.NoFocus)
        self._ui.tabular.itemChanged.connect(self._on_text_changed)

        self._module = None
        self._user_acts = None
        self._sentences = []
    
    def set_module(self, module):
        self._module = module
        self.update_sentences()
    
    def _get_user_act_string(self, user_act):
        """Creates a string representation of a given user act

        Arguments:
            user_act {UserAct} -- the user act to be displayed
        
        Returns:
            str -- how the given user act should be printed
        """
        out = user_act.type.value + '('
        if user_act.slot is not None:
            out += user_act.slot
            if user_act.value is not None:
                out += '=' + user_act.value
        out += ')'
        return out
    
    def _create_row(self, idx, sentence):
        """Draws the row for the specified index and sentence.
        
        Each row consists of a delete button on the left, a label stating
        whether or not the test case was recognised correctly and the
        test sentence as a line edit.
        This method assumes that the table widget has already been set
        to the correct size.

        Arguments:
            idx {int} -- the index of the test case
            sentence {str} -- the test case
        """
        delete_button = self._create_delete_button()
        delete_button.pressed.connect(lambda idx=idx: self._on_deleted(idx))
        self._ui.tabular.setCellWidget(idx, 0, delete_button)
        self._ui.tabular.setCellWidget(idx, 1, self._create_space(10))

        # check whether the test sentences are correctly analysed
        correct, user_acts = self._analyse_sentence(sentence)
        label = self._create_tick() if correct else self._create_cross()
        analysis = ', '.join([self._get_user_act_string(act) for act in user_acts])
        if len(user_acts) == 0:
            analysis = 'no acts found'
        label.setToolTip('Analysis: %s' % analysis)

        self._ui.tabular.setCellWidget(idx, 2, label)
        self._ui.tabular.setCellWidget(idx, 3, self._create_space(20))
        item = QTableWidgetItem(sentence)
        item.setBackground(QColor(96, 138, 255))
        self._ui.tabular.setItem(idx, 4, item)

    def _on_added(self):
        """callback function for the add button being clicked

        Adds a new test case to the list and automatically navigates to
        the corresponding line edit.
        """
        self._sentences.append('')
        self.tests_changed.emit(self._sentences[:])
        self.update_sentences()
        self._ui.tabular.setCurrentCell(len(self._sentences)-1, 4)
        self._ui.tabular.openPersistentEditor(self._ui.tabular.item(len(self._sentences)-1, 4))

    def _on_deleted(self, idx):
        """callback function for a delete button being clicked

        Deletes the test case at the given index.
        
        Arguments:
            idx {int} -- the index of the button being clicked,
                         i.e. the test case that should be deleted
        """
        self._sentences.pop(idx)
        self.tests_changed.emit(self._sentences[:])
        self.update_sentences()
    
    def _on_text_changed(self, item):
        """Callback function for when the user edited a test case
        
        Arguments:
            item {QTableWidgetItem} -- the item that was changed
        """
        self._sentences[item.row()] = item.text()
        self.tests_changed.emit(self._sentences[:])  # inform the edit widget
        self.update_sentences()
    
    def get_sentences(self):
        return self._sentences[:]
        
    def set_sentences(self, sentences, curr_user_acts):
        """Specifies the current user acts and their test cases.
        
        Function should be called by the edit widget when the UserActView has changed.

        Arguments:
            sentences {list[str]} -- the test cases
            curr_user_acts {frozenset[UserAct]} -- the current set of user acts
        """
        self._sentences = sentences[:]
        self._user_acts = curr_user_acts
        self.update_sentences()
    
    def _create_space(self, margin):
        """Creates a spacer item for the QTableWidget
        
        Arguments:
            margin {int} -- how much space the widget should take (in pixels)
        
        Returns:
            QWidget -- a widget with the specified width
                       that can be inserted in the QTableWidget
        """
        empty = QWidget(self)
        empty.setFixedWidth(margin)
        empty.setFixedHeight(10)
        empty.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return empty
    
    def _create_tick(self):
        """
        Returns:
            QLabel -- a label with a green tick inside
        """
        b = QLabel('✔', self)
        b.setFont(YN_FONT)
        b.setStyleSheet('QLabel { color: green; } QToolTip { color: #000; background-color: #bbffffcc; border: 1px solid black; }')
        return b
    
    def _create_cross(self):
        """
        Returns:
            QLabel -- a label with a red cross inside
        """
        b = QLabel('❌', self)
        b.setFont(YN_FONT)
        b.setStyleSheet('QLabel { color: red; } QToolTip { color: #000; background-color: #bbffffcc; border: 1px solid black; }')
        return b
    
    def _create_delete_button(self):
        """
        Returns:
            QPushButton -- a button with a minus
        """
        b = QPushButton('-', self)
        b.setFont(PM_FONT)
        b.setMaximumWidth(30)
        return b
    
    def _create_add_button(self):
        """
        Returns:
            QPushButton -- a button with a plus
        """
        b = QPushButton('+', self)
        b.setFont(PM_FONT)
        b.setMaximumWidth(30)
        return b
    
    def _analyse_sentence(self, sentence):
        """checks whether the given sentence is correctly analysed
        
        The current NLU module is called to analyse the given sentence.
        This method then checks whether the analysed user acts are correct,
        i.e. agree with current set of user acts (from the UserActView).

        Arguments:
            sentence {str} -- the test case
        
        Returns:
            bool -- whether or not the user acts are the same
            frozenset[UserAct] -- the set of user acts analysed by the NLU
        """
        result = self._module.forward(None, user_utterance=sentence)['user_acts']
        analysed_user_acts = frozenset(result)
        return analysed_user_acts == self._user_acts, analysed_user_acts
    
    def update_sentences(self):
        """Redraws the QTableWidget for the test cases.

        Resets the size of the QTableWidget, redraws the rows
        and creates the delete button.
        """
        self._ui.tabular.blockSignals(True)  # make sure the signals aren't called
        # reset the size of the QTableWidget
        self._ui.tabular.clear()
        self._ui.tabular.setRowCount(len(self._sentences)+1)
        self._ui.tabular.setColumnCount(5)

        # draw the rows for the test cases
        for i in range(len(self._sentences)):
            self._create_row(i, self._sentences[i])
        
        # draw the last row
        create_button = self._create_add_button()
        create_button.pressed.connect(self._on_added)
        self._ui.tabular.setCellWidget(len(self._sentences), 0, create_button)
        for i in range(1,5):
            self._ui.tabular.setCellWidget(len(self._sentences), i, self._create_space(1))
        
        # stretch only the last column
        self._ui.tabular.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._ui.tabular.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._ui.tabular.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._ui.tabular.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._ui.tabular.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)

        self._ui.tabular.blockSignals(False)
