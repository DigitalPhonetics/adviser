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

from enum import Enum


class MessageComponentType(Enum):
    STRING = 'string'
    ADVISER_CODE = 'adviser'
    PYTHON_CODE = 'python'


class MessageComponent:
    def __init__(self, component_type: MessageComponentType, value: str):
        self.component_type = component_type
        self.value = value

    def __repr__(self):
        return f'[{self.component_type.value}]"{self.value}"'
