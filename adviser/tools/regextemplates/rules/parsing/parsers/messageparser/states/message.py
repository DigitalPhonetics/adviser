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
    StateDescription, Configuration, TransitionWithoutAction, SimpleForwardDefaultTransition
from tools.regextemplates.rules.parsing.stack import AutomatonStack

from tools.regextemplates.rules.parsing.parsers.messageparser.data.messagecomponent import MessageComponent, MessageComponentType
from tools.regextemplates.rules.parsing.parsers.messageparser.states.statelist import \
    MessageState, EscapeState, AcceptState, CodeState


class _TransitionFromMessageToCode(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(MessageState(), '{'),
                                         Configuration(CodeState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        current_string = stack.get_current_content()
        stack.add_data(MessageComponent(MessageComponentType.STRING, current_string))
        stack.remove_level()
        stack.add_level()


class _TransitionFromMessageToAccept(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(MessageState(), '"'),
                                         Configuration(AcceptState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        current_string = stack.get_current_content()
        stack.add_data(MessageComponent(MessageComponentType.STRING, current_string))
        stack.remove_level()


class MessageStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(
            self, MessageState(), SimpleForwardDefaultTransition(MessageState()), [
                TransitionWithoutAction(Configuration(MessageState(), '\\'),
                                        Configuration(EscapeState(MessageState()), '')),
                _TransitionFromMessageToCode(),
                _TransitionFromMessageToAccept()
            ])
