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

from services.nlg.templates.data.expressions.constantexpression import ConstantExpression
from services.nlg.templates.parsing.configuration import \
    StateDescription, Configuration, TransitionWithoutAction, SimpleForwardDefaultTransition
from services.nlg.templates.parsing.parsers.constraintparser.states.statelist import \
    ValueState, ValueEscapeState, ConstraintEndState
from services.nlg.templates.parsing.stack import AutomatonStack


class _StringEndTransition(TransitionWithoutAction):
    def __init__(self):
        TransitionWithoutAction.__init__(self, Configuration(ValueState(), '"'),
                                         Configuration(ConstraintEndState(), ''))

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        current_string = stack.get_current_content()
        stack.add_data(ConstantExpression(current_string))
        stack.remove_level()
        ConstraintEndState.add_constraint_to_stack(stack)


class ValueStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(
            self, ValueState(), SimpleForwardDefaultTransition(ValueState()), [
                TransitionWithoutAction(Configuration(ValueState(), '\\'),
                                        Configuration(ValueEscapeState(), '')),
                _StringEndTransition()
            ])
