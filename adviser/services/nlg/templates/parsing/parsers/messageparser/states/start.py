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
from services.nlg.templates.parsing.parsers.messageparser.states.statelist import \
    StartState, MessageState
from services.nlg.templates.parsing.stack import AutomatonStack


class _InvalidStartTransition(DefaultTransition):
    def __init__(self):
        DefaultTransition.__init__(self, StartState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        raise ParsingError('An NLG message must start with a quotation mark!')

    def perform_stack_action(self, stack: AutomatonStack, input_configuration: Configuration):
        raise ParsingError('An NLG message must start with a quotation mark!')


class StartStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(self, StartState(), _InvalidStartTransition(), [
            TransitionWithAction(Configuration(StartState(), '"'),
                                 Configuration(MessageState(), ''),
                                 lambda stack: stack.add_level())
        ])
