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

from typing import List, Dict

from tools.regextemplates.rules.parsing.configuration import \
    State, StateDescription, Configuration, DefaultTransition, Transition
from tools.regextemplates.rules.parsing.stack import AutomatonStack
from tools.regextemplates.rules.parsing.exceptions import ParsingError


class ModifiedPushdownAutomaton:
    def __init__(self, start_state: State, accept_states: List[State],
                 state_descriptions: List[StateDescription]):
        self.start_state = start_state
        self.accept_states = accept_states
        self.state_descriptions = state_descriptions

        self.state_transition_mapping = self._create_state_transition_mapping()
        self.state_default_transition_mapping = self._create_state_default_transition_mapping()

        self.stack = AutomatonStack()

    def _create_state_transition_mapping(self) -> Dict[State, Dict[str, Transition]]:
        state_transition_mapping = {}
        for state_description in self.state_descriptions:
            input_state = state_description.default_state
            if input_state not in state_transition_mapping:
                state_transition_mapping[input_state] = {}
            for transition in state_description.transitions:
                input_char = transition.input_configuration.character
                state_transition_mapping[input_state][input_char] = transition
        return state_transition_mapping

    def _create_state_default_transition_mapping(self) -> Dict[State, DefaultTransition]:
        state_default_transition_mapping = {}
        for state_description in self.state_descriptions:
            state_default_transition_mapping[state_description.default_state] = \
                state_description.default_transition
        return state_default_transition_mapping

    def parse(self, input_tape: str) -> List[object]:
        self.stack.clear()
        current_state = self.start_state
        input_tape_index = 0

        for input_char in input_tape:
            try:
                configuration = Configuration(current_state, input_char)
                transition = self._find_transition(configuration)
                current_state = self._apply_transition(transition, configuration)
                input_tape_index += 1
            except ParsingError as error:
                print('State:', current_state.name)
                print('Index:', input_tape_index)
                print('Original Input:', input_tape)
                raise error

        if current_state not in self.accept_states:
            print('State:', current_state.name)
            raise ParsingError(f'Parser was not in a final state after the input tape was read.')

        return self.stack.data_stack[:]

    def _apply_transition(self, transition: Transition,
                          input_configuration: Configuration) -> State:
        transition.perform_stack_action(self.stack, input_configuration)
        output_configuration = transition.get_output_configuration(input_configuration)
        self.stack.add_char(output_configuration.character)
        return output_configuration.state

    def _find_transition(self, configuration: Configuration):
        if configuration.state not in self.state_transition_mapping or \
            configuration.character not in self.state_transition_mapping[configuration.state]:
            return self._find_default_transition(configuration.state)
        return self.state_transition_mapping[configuration.state][configuration.character]

    def _find_default_transition(self, current_state: State):
        if current_state not in self.state_default_transition_mapping:
            raise ParsingError(f'No default transition found for state {current_state.name}.')
        return self.state_default_transition_mapping.get(current_state, None)
