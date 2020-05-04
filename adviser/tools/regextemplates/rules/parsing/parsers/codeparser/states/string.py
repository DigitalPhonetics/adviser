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

from tools.regextemplates.rules.data.expressions.constantexpression import ConstantExpression
from tools.regextemplates.rules.parsing.parsers.codeparser.states.statelist import \
    StringState, StringEscapeState, ExpressionEndState


class _StringEndTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(StringState(), '"'),
                                         Configuration(ExpressionEndState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        current_string = stack.get_current_content()
        stack.add_data(ConstantExpression(current_string))
        stack.remove_level()
        stack.add_level()


class StringStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(
            self, StringState(), SimpleForwardDefaultTransition(StringState()), [
                TransitionWithoutAction(Configuration(StringState(), '\\'),
                                        Configuration(StringEscapeState(), '')),
                _StringEndTransition()
            ])
