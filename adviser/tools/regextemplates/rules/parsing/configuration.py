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

from typing import Callable, List

from tools.regextemplates.rules.parsing.stack import AutomatonStack


class State:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, State) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class Configuration:
    def __init__(self, state: State, character: str):
        self.state = state
        self.character = character


class Transition:
    def __init__(self, input_configuration: Configuration):
        self.input_configuration = input_configuration

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        raise NotImplementedError()

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        raise NotImplementedError()


class DefaultTransition(Transition):
    def __init__(self, state: State):
        Transition.__init__(self, Configuration(state, None))


class StateDescription:
    def __init__(self, default_state: State, default_transition: DefaultTransition,
                 transitions: List[Transition]):
        self.default_state = default_state
        self.default_transition = default_transition
        self.transitions = transitions


class SimpleForwardDefaultTransition(DefaultTransition):
    def __init__(self, state: State):
        DefaultTransition.__init__(self, state)

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        return Configuration(input_configuration.state, input_configuration.character)

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class TransitionWithoutAction(Transition):
    def __init__(self, input_configuration: Configuration, output_configuration: Configuration):
        Transition.__init__(self, input_configuration)
        self.output_configuration = output_configuration

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        return self.output_configuration

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class TransitionWithAction(Transition):
    def __init__(self, input_configuration: Configuration, output_configuration: Configuration,
                 action: Callable[[AutomatonStack], None]):
        Transition.__init__(self, input_configuration)
        self.output_configuration = output_configuration
        self.action = action

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        return self.output_configuration

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        self.action(stack)
