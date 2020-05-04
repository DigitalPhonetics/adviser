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
    StateDescription, Configuration, TransitionWithoutAction, DefaultTransition
from services.nlg.templates.parsing.exceptions import ParsingError
from services.nlg.templates.parsing.parsers.codeparser.states.statelist import \
    MemberState, AcceptState, ExpressionState, ExpressionEndState
from services.nlg.templates.parsing.stack import AutomatonStack


class _MemberDefaultTransition(DefaultTransition):
    def __init__(self):
        DefaultTransition.__init__(self, MemberState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        if not input_configuration.character.isalpha() and input_configuration.character != '_':
            raise ParsingError(f'Non-alpha character "{input_configuration.character}" detected.')
        return Configuration(input_configuration.state, input_configuration.character)

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class _MemberWhitespaceTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(MemberState(), ' '),
                                         Configuration(ExpressionEndState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        MemberState.check_stack(stack)
        stack.add_data(stack.get_current_content())
        MemberState.add_member_to_stack(stack)
        stack.remove_level()
        stack.add_level()


class _MemberCommaTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(MemberState(), ','),
                                         Configuration(ExpressionState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        MemberState.check_stack(stack)
        stack.add_data(stack.get_current_content())
        MemberState.add_member_to_stack(stack)
        stack.remove_level()
        stack.add_level()


class _TransitionFromMemberToAccept(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(MemberState(), '$'),
                                         Configuration(AcceptState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        MemberState.check_stack(stack)
        stack.add_data(stack.get_current_content())
        MemberState.add_member_to_stack(stack)
        stack.remove_level()
        AcceptState.check_stack(stack)


class _TransitionFromMemberToCloseFunction(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(MemberState(), ')'),
                                         Configuration(ExpressionEndState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        MemberState.check_stack(stack)
        stack.add_data(stack.get_current_content())
        MemberState.add_member_to_stack(stack)
        stack.remove_level()
        ExpressionEndState.add_function_to_stack(stack)


class MemberStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(self, MemberState(), _MemberDefaultTransition(), [
            _TransitionFromMemberToAccept(),
            _MemberCommaTransition(),
            _MemberWhitespaceTransition(),
            _TransitionFromMemberToCloseFunction()
        ])
