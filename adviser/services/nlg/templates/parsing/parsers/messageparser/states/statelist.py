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

from services.nlg.templates.parsing.configuration import State


class StartState(State):
    def __init__(self):
        State.__init__(self, 'START')

class AcceptState(State):
    def __init__(self):
        State.__init__(self, 'ACCEPT')

class MessageState(State):
    def __init__(self):
        State.__init__(self, 'MESSAGE')

class CodeState(State):
    def __init__(self):
        State.__init__(self, 'CODE')

class AdviserState(State):
    def __init__(self):
        State.__init__(self, 'CODE_ADVISER')

class PythonState(State):
    def __init__(self):
        State.__init__(self, 'CODE_PYTHON')

class PythonClosingBraceState(State):
    def __init__(self):
        State.__init__(self, 'CODE_PYTHON_CLOSING_BRACE')

class EscapeState(State):
    def __init__(self, parent_state: State):
        State.__init__(self, 'ESCAPE')
        self.parent_state = parent_state

class CodeStringState(State):
    def __init__(self, parent_state: State):
        State.__init__(self, 'CODE_STRING')
        self.parent_state = parent_state
