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

from tools.regextemplates.rules.parsing.configuration import \
    State, StateDescription, Configuration, DefaultTransition, TransitionWithoutAction
from tools.regextemplates.rules.parsing.exceptions import ParsingError
from tools.regextemplates.rules.parsing.stack import AutomatonStack

from tools.regextemplates.rules.parsing.parsers.messageparser.states.statelist import \
    ProbabilityState, AcceptState


class _ProbabilityDefaultTransition(DefaultTransition):
    def __init__(self):
        DefaultTransition.__init__(self, ProbabilityState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        if input_configuration.character != '.' and not input_configuration.isdigit():
            raise ParsingError('Expected a number as probability/weight.')

    def perform_stack_action(self, stack: AutomatonStack, input_configuration: Configuration):
        current_level = 
        if input_configuration.character != '$':
            raise ParsingError('Expected a number as probability/weight.')


class _ProbabilityEndTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(ProbabilityState(), '$'),
                                         Configuration(ProbabilityState(), ''))
    
    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        content = stack.get_current_content()
        try:
            stack.add_data(MessageComponent(MessageComponentType.PROBABILITY,
                                            float(current_string)))
            stack.remove_level()
        except ValueError:
            raise ParsingError(f'{content} could not be interpreted as a probability/weight.')


class ProbabilityStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(self, ProbabilityState(), _ProbabilityDefaultTransition(), [
            _ProbabilityEndTransition()
        ])
