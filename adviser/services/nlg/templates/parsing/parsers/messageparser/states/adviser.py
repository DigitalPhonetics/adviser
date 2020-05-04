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
    StateDescription, Configuration, TransitionWithoutAction, SimpleForwardDefaultTransition
from services.nlg.templates.parsing.parsers.messageparser.data.messagecomponent import MessageComponent, \
    MessageComponentType
from services.nlg.templates.parsing.parsers.messageparser.states.statelist import \
    AdviserState, MessageState, CodeStringState
from services.nlg.templates.parsing.stack import AutomatonStack


class _AdviserEndTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(AdviserState(), '}'),
                                         Configuration(MessageState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        current_string = stack.get_current_content()
        stack.add_data(MessageComponent(MessageComponentType.ADVISER_CODE, current_string))
        stack.remove_level()
        stack.add_level()


class AdviserStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(
            self, AdviserState(), SimpleForwardDefaultTransition(AdviserState()), [
                TransitionWithoutAction(Configuration(AdviserState(), '"'),
                                        Configuration(CodeStringState(AdviserState()), '"')),
                _AdviserEndTransition()
            ])
