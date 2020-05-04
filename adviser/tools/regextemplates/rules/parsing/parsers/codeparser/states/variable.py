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

from tools.regextemplates.rules.parsing.configuration import \
    StateDescription, Configuration, TransitionWithoutAction, DefaultTransition
from tools.regextemplates.rules.parsing.exceptions import ParsingError
from tools.regextemplates.rules.parsing.stack import AutomatonStack

from tools.regextemplates.rules.data.expressions.variableexpression import VariableExpression
from tools.regextemplates.rules.parsing.parsers.codeparser.data.functiondeclaration import FunctionDeclaration
from tools.regextemplates.rules.parsing.parsers.codeparser.states.statelist import VariableState, \
    AcceptState, ExpressionState, MemberState, OptionalExpressionState, ExpressionEndState


class _VariableDefaultTransition(DefaultTransition):
    def __init__(self):
        DefaultTransition.__init__(self, VariableState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        if not input_configuration.character.isalpha() and input_configuration.character != '_':
            raise ParsingError(f'Non-alpha character "{input_configuration.character}" detected.')
        return Configuration(input_configuration.state, input_configuration.character)

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class _VariableWhitespaceTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(VariableState(), ' '),
                                         Configuration(ExpressionEndState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        stack.add_data(VariableExpression(stack.get_current_content()))
        stack.remove_level()
        stack.add_level()


class _VariableCommaTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(VariableState(), ','),
                                         Configuration(ExpressionState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        stack.add_data(VariableExpression(stack.get_current_content()))
        stack.remove_level()
        stack.add_level()


class _TransitionFromVariableToAccept(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(VariableState(), '$'),
                                         Configuration(AcceptState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        stack.add_data(VariableExpression(stack.get_current_content()))
        stack.remove_level()
        AcceptState.check_stack(stack)


class _TransitionFromVariableToMember(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(VariableState(), '.'),
                                         Configuration(MemberState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        stack.add_data(VariableExpression(stack.get_current_content()))
        stack.remove_level()
        stack.add_level()


class _TransitionFromVariableToOpenFunction(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(VariableState(), '('),
                                         Configuration(OptionalExpressionState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        stack.add_data(FunctionDeclaration(stack.get_current_content()))
        stack.remove_level()
        stack.add_level()


class _TransitionFromVariableToCloseFunction(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(VariableState(), ')'),
                                         Configuration(ExpressionEndState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        stack.add_data(VariableExpression(stack.get_current_content()))
        stack.remove_level()
        ExpressionEndState.add_function_to_stack(stack)


class VariableStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(self, VariableState(), _VariableDefaultTransition(), [
            _TransitionFromVariableToAccept(),
            _VariableCommaTransition(),
            _TransitionFromVariableToMember(),
            _TransitionFromVariableToOpenFunction(),
            _VariableWhitespaceTransition(),
            _TransitionFromVariableToCloseFunction()
        ])
