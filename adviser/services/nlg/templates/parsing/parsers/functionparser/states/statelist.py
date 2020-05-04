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
from services.nlg.templates.parsing.stack import AutomatonStack


# from services.nlg.templates.parsing.parsers.functionparser.data.commands.function import Function


class FunctionNameState(State):
    def __init__(self):
        State.__init__(self, 'FUNCTION_NAME')


class ArgumentStartState(State):
    def __init__(self):
        State.__init__(self, 'ARGUMENT_START')

class ArgumentState(State):
    def __init__(self):
        State.__init__(self, 'ARGUMENT')

class FreeArgumentState(State):
    def __init__(self):
        State.__init__(self, 'FREE_ARGUMENT')

class FreeArgumentStartState(State):
    def __init__(self):
        State.__init__(self, 'FREE_ARGUMENT_START')

class RequiredArgumentState(State):
    def __init__(self):
        State.__init__(self, 'REQUIRED_ARGUMENT')

class AcceptState(State):
    def __init__(self):
        State.__init__(self, 'ACCEPT')

    @staticmethod
    def add_function_to_stack(stack: AutomatonStack):
        pass
        """arguments = []
        while len(stack.data_stack) > 1:
            arguments.insert(0, stack.pop_data())
        function_declaration = stack.pop_data()
        stack.add_data(Function(function_declaration, arguments))"""
