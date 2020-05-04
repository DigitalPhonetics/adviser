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

from tools.regextemplates.rules.parsing.configuration import StateDescription, \
    Configuration, TransitionWithoutAction, TransitionWithAction, DefaultTransition
from tools.regextemplates.rules.parsing.exceptions import ParsingError
from tools.regextemplates.rules.parsing.stack import AutomatonStack

from tools.regextemplates.rules.parsing.parsers.codeparser.data.functiondeclaration import FunctionDeclaration
from tools.regextemplates.rules.parsing.parsers.codeparser.states.statelist import \
    ExpressionEndState, AcceptState, ExpressionState


class _ExpressionEndDefaultTransition(DefaultTransition):
    def __init__(self):
        DefaultTransition.__init__(self, ExpressionEndState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        ParsingError(f'Unexpected character "{input_configuration.character}" after '
                     f'the expression ended.')

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class _ConversionFromVariableToFunctionName(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(ExpressionEndState(), '('),
                                         Configuration(ExpressionState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        variable = stack.pop_data()
        stack.add_data(FunctionDeclaration(variable.name))


class ExpressionEndStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(self, ExpressionEndState(), _ExpressionEndDefaultTransition(), [
            TransitionWithoutAction(Configuration(ExpressionEndState(), ' '),
                                    Configuration(ExpressionEndState(), '')),
            TransitionWithAction(Configuration(ExpressionEndState(), '$'),
                                 Configuration(AcceptState(), ''),
                                 AcceptState.check_stack),
            TransitionWithoutAction(Configuration(ExpressionEndState(), ','),
                                    Configuration(ExpressionState(), '')),
            _ConversionFromVariableToFunctionName(),
            TransitionWithAction(Configuration(ExpressionEndState(), ')'),
                                 Configuration(ExpressionEndState(), ''),
                                 ExpressionEndState.add_function_to_stack)
        ])
