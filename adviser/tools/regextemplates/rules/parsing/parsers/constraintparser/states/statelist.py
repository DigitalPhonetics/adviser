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

from tools.regextemplates.rules.parsing.configuration import State
from tools.regextemplates.rules.parsing.stack import AutomatonStack
from tools.regextemplates.rules.parsing.parsers.constraintparser.data.constraint import Constraint


class ConstraintStartState(State):
    def __init__(self):
        State.__init__(self, 'CONSTRAINT_START')

class VariableState(State):
    def __init__(self):
        State.__init__(self, 'VARIABLE')

class VariableEndState(State):
    def __init__(self):
        State.__init__(self, 'VARIABLE_END')

class OperatorState(State):
    def __init__(self):
        State.__init__(self, 'OPERATOR')

class OperatorEndState(State):
    def __init__(self):
        State.__init__(self, 'OPERATOR_END')

class ValueState(State):
    def __init__(self):
        State.__init__(self, 'VALUE')

class ValueEscapeState(State):
    def __init__(self):
        State.__init__(self, 'VALUE_ESCAPE')

class ConstraintEndState(State):
    def __init__(self):
        State.__init__(self, 'CONSTRAINT_END')

    @staticmethod
    def add_constraint_to_stack(stack: AutomatonStack):
        value = stack.pop_data()
        variable = stack.pop_data()

        stack.add_data(Constraint(variable, value))
