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
    StateDescription, Configuration, Transition, SimpleForwardDefaultTransition
from tools.regextemplates.rules.parsing.stack import AutomatonStack

from tools.regextemplates.rules.parsing.parsers.messageparser.states.statelist import \
    CodeStringState, EscapeState


class CodeStringEndTransition(Transition):
    def __init__(self):
        Transition.__init__(self, Configuration(CodeStringState(None), '"'))

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        assert isinstance(input_configuration.state, CodeStringState)
        return Configuration(input_configuration.state.parent_state, '"')

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class TransitionFromCodeStringToEscape(Transition):
    def __init__(self):
        Transition.__init__(self, Configuration(CodeStringState(None), '\\'))

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        output_state = EscapeState(input_configuration.state)
        return Configuration(output_state, '\\')

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class CodeStringStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(
            self, CodeStringState(None), SimpleForwardDefaultTransition(CodeStringState(None)), [
                TransitionFromCodeStringToEscape(),
                CodeStringEndTransition()
            ])
