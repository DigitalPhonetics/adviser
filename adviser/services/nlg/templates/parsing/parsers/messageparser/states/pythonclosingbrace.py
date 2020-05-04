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
    StateDescription, Configuration, DefaultTransition, TransitionWithoutAction
from services.nlg.templates.parsing.exceptions import ParsingError
from services.nlg.templates.parsing.parsers.messageparser.data.messagecomponent import MessageComponent, \
    MessageComponentType
from services.nlg.templates.parsing.parsers.messageparser.states.statelist import \
    PythonClosingBraceState, MessageState
from services.nlg.templates.parsing.stack import AutomatonStack


class _InvalidSingleClosingBraceTransition(DefaultTransition):
    """This is called whenever a closing brace was not followed by another closing brace"""

    def __init__(self):
        DefaultTransition.__init__(self, PythonClosingBraceState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        raise ParsingError('The python code environment requires two closing braces!')

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        raise ParsingError('The python code environment requires two closing braces!')


class PythonEndTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(PythonClosingBraceState(), '}'),
                                         Configuration(MessageState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        current_string = stack.get_current_content()
        stack.add_data(MessageComponent(MessageComponentType.PYTHON_CODE, current_string))
        stack.remove_level()
        stack.add_level()


class PythonClosingBraceStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(
            self, PythonClosingBraceState(), _InvalidSingleClosingBraceTransition(), [
                PythonEndTransition()
            ])
