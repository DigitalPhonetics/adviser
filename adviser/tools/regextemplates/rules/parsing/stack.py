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

from tools.regextemplates.rules.parsing.exceptions import ParsingError


class AutomatonStack:
    def __init__(self):
        # self.char_stack = []  # the automaton's stack
        self.data_stack = []  # the stack in which custom data structures can be stored
        self.levels = [[]]  # multiple automaton stacks are possible here

    def add_char(self, stack_char: str):
        if not self.levels:
            raise ParsingError('No more levels left on the stack')
        self.levels[-1].append(stack_char)

    def add_data(self, data: object):
        self.data_stack.append(data)

    def pop_data(self) -> object:
        return self.data_stack.pop(-1)

    def fetch_data(self) -> object:
        return self.data_stack[-1]

    def add_level(self):
        self.levels.append([])

    def get_current_content(self) -> str:
        if not self.levels:
            raise ParsingError('No more levels left on the stack')
        return ''.join(self.levels[-1])

    def remove_level(self):
        if not self.levels:
            raise ParsingError('No more levels to remove from the stack')
        self.levels.pop()

    def clear(self):
        self.data_stack = []
        self.levels = [[]]
