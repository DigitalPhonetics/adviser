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
from PyQt5.QtWidgets import QWidget, QPushButton, QComboBox, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
from utils import UserAct, UserActionType


INTENT_IDX = {
    UserActionType.Bye: 0,
    UserActionType.Hello: 1,
    UserActionType.Inform: 2,
    UserActionType.Request: 3
}

INTENTS = [
    UserActionType.Bye,
    UserActionType.Hello,
    UserActionType.Inform,
    UserActionType.Request
]

NICE_FONT = QFont('Sans Serif', 12)
PM_FONT = QFont('Consolas', 12)
PM_FONT.setBold(True)


class UserActView(QWidget):
    """Widget that displays the user acts for which the user can write test cases
    
    Attributes:
        user_acts_changed {pyqtSignal} -- emited when a user act was added, deleted or edited
        _user_acts {frozenset[UserAct]} -- the current user acts inside this widget
        _domain {Domain} -- the domain of the NLU module (which specifies possible user acts)
    """

    user_acts_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._ui = uic.loadUi(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui/useractview.ui'), self)
        
        self._ui.tabular.setFocusPolicy(Qt.NoFocus)

        self._domain = None
        self._user_acts = [ self._create_dummy_user_act() ]
    
    def set_domain(self, domain):
        self._domain = domain
        self.update_user_acts()
    
    def _get_intent_slots(self, user_act):
        if user_act.type == UserActionType.Inform:
            return list(self._domain.get_informable_slots())
        elif user_act.type == UserActionType.Request:
            return list(self._domain.get_requestable_slots())
        else:
            return []
    
    def _get_slot_values(self, user_act, slot):
        if user_act.type == UserActionType.Inform:
            return self._domain.get_possible_values(slot)
        else:
            return []
    
    def _create_delete_button(self):
        b = QPushButton('-', self)
        b.setFont(PM_FONT)
        b.setMaximumWidth(30)
        return b
    
    def _create_add_button(self):
        b = QPushButton('+', self)
        b.setFont(PM_FONT)
        b.setMaximumWidth(30)
        return b
    
    def _create_label(self, text):
        b = QLabel(text, self)
        b.setFont(NICE_FONT)
        return b
    
    def _create_space(self, margin):
        #hs = QSpacerItem(margin, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        #return hs
        empty = QWidget(self)
        empty.setFixedWidth(margin)
        empty.setFixedHeight(10)
        empty.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return empty
    
    def _create_combobox(self, items):
        cbox = QComboBox(self)
        cbox.addItems(items)
        cbox.setFont(NICE_FONT)
        cbox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        return cbox
    
    def _create_dummy_user_act(self):
        dummy = UserAct()
        dummy.type = UserActionType.Hello
        dummy.score = 1.0
        return dummy
    
    def _create_row(self, idx, user_act):
        delete_button = self._create_delete_button()
        delete_button.pressed.connect(lambda idx=idx: self._on_act_deleted(idx))
        self._ui.tabular.setCellWidget(idx, 0, delete_button)
        self._ui.tabular.setCellWidget(idx, 1, self._create_space(10))
        self._ui.tabular.setCellWidget(idx, 2, self._create_label('Intent:'))
        self._ui.tabular.setCellWidget(idx, 3, self._create_space(20))

        intent = self._create_combobox(['Bye', 'Hello', 'Inform', 'Repeat', 'Request'])
        intent.setCurrentIndex(INTENT_IDX[user_act.type])
        intent.currentIndexChanged.connect(lambda i_idx, act_idx=idx: self._on_intent_changed(act_idx, i_idx))
        self._ui.tabular.setCellWidget(idx, 4, intent)

        slots = self._get_intent_slots(user_act)
        if user_act.slot is not None:
            self._ui.tabular.setCellWidget(idx, 5, self._create_space(40))
            self._ui.tabular.setCellWidget(idx, 6, self._create_label('Slot:'))
            self._ui.tabular.setCellWidget(idx, 7, self._create_space(20))
            slot_cb = self._create_combobox(slots)
            slot_cb.setCurrentIndex(slots.index(user_act.slot))
            slot_cb.currentIndexChanged.connect(lambda s_idx, act_idx=idx: self._on_slot_changed(act_idx, s_idx))
            self._ui.tabular.setCellWidget(idx, 8, slot_cb)

            values = self._get_slot_values(user_act, user_act.slot)
            if len(values) > 0:
                value = user_act.value
                self._ui.tabular.setCellWidget(idx, 9, self._create_space(40))
                self._ui.tabular.setCellWidget(idx, 10, self._create_label('Value:'))
                self._ui.tabular.setCellWidget(idx, 11, self._create_space(20))
                value_cb = self._create_combobox(values)
                value_cb.setCurrentIndex(values.index(value))
                value_cb.currentIndexChanged.connect(lambda v_idx, act_idx=idx: self._on_value_changed(act_idx, v_idx))
                self._ui.tabular.setCellWidget(idx, 12, value_cb)
    
    def update_user_acts(self):
        self._ui.tabular.clear()
        self._ui.tabular.setRowCount(len(self._user_acts)+1)
        self._ui.tabular.setColumnCount(13)
        for i in range(len(self._user_acts)):
            self._create_row(i, self._user_acts[i])
        create_button = self._create_add_button()
        create_button.pressed.connect(self._on_act_added)
        self._ui.tabular.setCellWidget(len(self._user_acts), 0, create_button)

        self._ui.tabular.resizeColumnsToContents()
        self._ui.tabular.resizeRowsToContents()
    
    def _on_act_added(self):
        self._user_acts.append(self._create_dummy_user_act())
        self.user_acts_changed.emit(self._user_acts[:])
        self.update_user_acts()
    
    def _on_act_deleted(self, idx):
        self._user_acts.pop(idx)
        self.user_acts_changed.emit(self._user_acts[:])
        self.update_user_acts()
    
    def _on_intent_changed(self, act_idx, intent_idx):
        new_user_act = UserAct()
        new_user_act.type = INTENTS[intent_idx]
        new_user_act.score = 1.0
        
        slots = self._get_intent_slots(new_user_act)
        if len(slots) > 0:
            slot = slots[0]
            new_user_act.slot = slot
            values = self._get_slot_values(new_user_act, slot)
            if len(values) > 0:
                new_user_act.value = values[0]
            else:
                new_user_act.value = None
        self._user_acts[act_idx] = new_user_act
        self.user_acts_changed.emit(self._user_acts[:])
        self.update_user_acts()
    
    def _on_slot_changed(self, act_idx, slot_idx):
        new_user_act = UserAct()
        new_user_act.type = self._user_acts[act_idx].type
        new_user_act.score = 1.0
        
        slots = self._get_intent_slots(new_user_act)
        if len(slots) > 0:
            slot = slots[slot_idx]
            new_user_act.slot = slot
            values = self._get_slot_values(new_user_act, slot)
            if len(values) > 0:
                new_user_act.value = values[0]
            else:
                new_user_act.value = None
        self._user_acts[act_idx] = new_user_act
        self.user_acts_changed.emit(self._user_acts[:])
        self.update_user_acts()
    
    def _on_value_changed(self, act_idx, value_idx):
        new_user_act = UserAct()
        new_user_act.type = self._user_acts[act_idx].type
        new_user_act.score = 1.0
        
        slot = self._user_acts[act_idx].slot
        values = self._get_slot_values(new_user_act, slot)
        new_user_act.slot = slot
        new_user_act.value = values[value_idx]
        self._user_acts[act_idx] = new_user_act
        self.user_acts_changed.emit(self._user_acts[:])
        self.update_user_acts()
    
    def get_user_acts(self):
        return self._user_acts[:]
        
    def set_user_acts(self, user_acts):
        self._user_acts = user_acts[:]
        self.update_user_acts()
