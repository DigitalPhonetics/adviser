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

from tools.regextemplates.rules.parsing.parsers.functionparser.states.statelist import \
    ArgumentState, RequiredArgumentState, AcceptState
from tools.regextemplates.rules.parsing.parsers.functionparser.data.argument import Argument


class _ArgumentDefaultTransition(DefaultTransition):
    def __init__(self):
        DefaultTransition.__init__(self, ArgumentState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        if not input_configuration.character.isalpha() and input_configuration.character != '_':
            raise ParsingError(f'Non-alpha character "{input_configuration.character}" detected.')
        return Configuration(input_configuration.state, input_configuration.character)

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class _ArgumentCommaTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(ArgumentState(), ','),
                                         Configuration(RequiredArgumentState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        current_content = stack.get_current_content()
        if not current_content:
            raise ParsingError('A variable name was expected!')
        stack.add_data(Argument(current_content))
        stack.remove_level()


class _TransitionFromArgumentToAccept(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(ArgumentState(), ')'),
                                         Configuration(AcceptState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        current_content = stack.get_current_content()
        if not current_content:
            raise ParsingError('A variable name was expected!')
        stack.add_data(Argument(current_content))
        stack.remove_level()
        AcceptState.add_function_to_stack(stack)


class ArgumentStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(
            self, ArgumentState(), _ArgumentDefaultTransition(), [
                _ArgumentCommaTransition(),
                _TransitionFromArgumentToAccept()
            ])
