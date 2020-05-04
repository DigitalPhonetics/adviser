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

from services.nlg.templates.parsing.configuration import \
    StateDescription, Configuration, DefaultTransition, TransitionWithAction
from services.nlg.templates.parsing.exceptions import ParsingError
from services.nlg.templates.parsing.parsers.constraintparser.states.statelist import \
    VariableEndState, OperatorState
from services.nlg.templates.parsing.stack import AutomatonStack


class _InvalidOperatorTransition(DefaultTransition):
    def __init__(self):
        DefaultTransition.__init__(self, VariableEndState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        raise ParsingError('Expected an operator for the constraint (e.g. "=")!')

    def perform_stack_action(self, stack: AutomatonStack, input_configuration: Configuration):
        raise ParsingError('Expected an operator for the constraint (e.g. "=")!')


class VariableEndStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(self, VariableEndState(), _InvalidOperatorTransition(), [
            TransitionWithAction(Configuration(VariableEndState(), '='),
                                 Configuration(OperatorState(), ''),
                                 lambda stack: stack.add_level())
        ])
