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

from services.nlg.templates.data.expressions.memberexpression import MemberExpression
from services.nlg.templates.parsing.configuration import State
from services.nlg.templates.parsing.exceptions import ParsingError
from services.nlg.templates.parsing.parsers.codeparser.data.functiondeclaration import FunctionDeclaration
from services.nlg.templates.parsing.stack import AutomatonStack


class ExpressionState(State):
    def __init__(self):
        State.__init__(self, 'EXPRESSION')


class OptionalExpressionState(State):
    def __init__(self):
        State.__init__(self, 'OPTIONAL_EXPRESSION')

class AcceptState(State):
    def __init__(self):
        State.__init__(self, 'ACCEPT')

    @staticmethod
    def check_stack(stack: AutomatonStack):
        if len(stack.data_stack) != 1:
            ParsingError('At the end of the code environment, there must be exactly'
                         'one element on the stack!')

class StringState(State):
    def __init__(self):
        State.__init__(self, 'STRING')

class StringEscapeState(State):
    def __init__(self):
        State.__init__(self, 'STRING_ESCAPE')

class VariableState(State):
    def __init__(self):
        State.__init__(self, 'VARIABLE')

class MemberState(State):
    def __init__(self):
        State.__init__(self, 'MEMBER')

    @staticmethod
    def add_member_to_stack(stack: AutomatonStack):
        assert len(stack.data_stack) >= 2

        member = stack.pop_data()
        variable = stack.pop_data()
        stack.add_data(MemberExpression(variable.name, member))

    @staticmethod
    def check_stack(stack: AutomatonStack):
        if stack.get_current_content() == '':
            ParsingError('Empty member variable detected')

class ExpressionEndState(State):
    def __init__(self):
        State.__init__(self, 'EXPRESSION_END')

    @staticmethod
    def add_function_to_stack(stack: AutomatonStack):
        arguments = []
        if not stack.data_stack:
            raise ParsingError('Detected a closing parantheses without an opening one')
        stack_content = stack.pop_data()
        while not isinstance(stack_content, FunctionDeclaration):
            arguments.insert(0, stack_content)
            if not stack.data_stack:
                raise ParsingError('Detected a closing parantheses without an opening one')
            stack_content = stack.pop_data()
        stack.add_data(stack_content.create_function(arguments))
