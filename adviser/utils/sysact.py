###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
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

"""This module provides the necessary classes for a system action."""

from enum import Enum
from typing import Dict, List


class SysActionType(Enum):
    """The type for a system action as used in :class:`SysAct`."""

    Welcome = 'welcomemsg'
    InformByName = 'inform_byname'
    InformByAlternatives = 'inform_alternatives'
    Request = 'request'
    Confirm = 'confirm'
    Select = 'select'
    RequestMore = 'reqmore'
    Bad = 'bad'
    Bye = 'closingmsg'
    ConfirmRequest = 'confreq'


class SysAct(object):

    def __init__(self, act_type: SysActionType = None, slot_values: Dict[str, List[str]] = None):
        """
        The class for a system action as used in the dialog.

        Args:
            act_type (SysActionType): The type of the system action.
            slot_values (Dict[str, List[str]]): A mapping of ``slot -> value`` to which the system
                action refers depending on the action type - might be ``None``.

        """
        self.type = act_type
        self.slot_values = slot_values if slot_values is not None else {}

    def __repr__(self):
        return f"""SysAct(act_type={self.type}
            {f", {self._slot_value_dict_to_str(self.slot_values)}"
            if self._slot_value_dict_to_str(self.slot_values) else ""})"""

    def __str__(self):
        if self.type is not None:
            return self.type.value + \
                   '(' + \
                   self._slot_value_dict_to_str(self.slot_values) + \
                   ')'
        else:
            return 'SysAct(act_type=' + self.type + ', ' + \
                    self._slot_value_dict_to_str(self.slot_values) + \
                    ')'

    def add_value(self, slot: str, value=None):
        """ Add a value (or just a slot, if value=None) to the system act """
        if slot not in self.slot_values:
            self.slot_values[slot] = []
        if value is not None:
            self.slot_values[slot].append(value)

    def get_values(self, slot) -> list:
        """ Return all values for slot

        Returns:
            A list of values for slot or an empy list if there was no value
            specified for the given slot
        """
        if slot not in self.slot_values:
            return []
        else:
            return self.slot_values[slot]

    def __eq__(self, other):
        return (self.type == other.type and
                self.slot_values == other.slot_values)

    def _slot_value_dict_to_str(self, slot_value_dict):
        """ convert dictionary to slot1=value1, slot2=value2, ... string """
        stringrep = []
        for slot in slot_value_dict:
            if slot_value_dict[slot]:
                if isinstance(slot_value_dict, list):
                    # there are values specified for slot, add them
                    for value in slot_value_dict[slot]:
                        if value is not None:
                            stringrep.append('{}="{}"'.format(slot, value))
                else:
                    if slot_value_dict[slot] is not None:
                        stringrep.append('{}="{}"'.format(slot, slot_value_dict[slot]))
            else:
                # slot without value
                stringrep.append(slot)
        return ','.join(stringrep)
