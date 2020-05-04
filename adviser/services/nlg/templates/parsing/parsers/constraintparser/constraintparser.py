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

import os
import sys

head_location = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..')) # main folder of adviser
sys.path.append(head_location)

from services.nlg.templates.parsing.automaton import ModifiedPushdownAutomaton

from services.nlg.templates.parsing.parsers.constraintparser.states.statelist import ConstraintStartState, \
    ConstraintEndState

from services.nlg.templates.parsing.parsers.constraintparser.states.constraintend import ConstraintEndStateDescription
from services.nlg.templates.parsing.parsers.constraintparser.states.constraintstart import \
    ConstraintStartStateDescription
from services.nlg.templates.parsing.parsers.constraintparser.states.operator import OperatorStateDescription
from services.nlg.templates.parsing.parsers.constraintparser.states.operatorend import OperatorEndStateDescription
from services.nlg.templates.parsing.parsers.constraintparser.states.value import ValueStateDescription
from services.nlg.templates.parsing.parsers.constraintparser.states.valueescape import ValueEscapeStateDescription
from services.nlg.templates.parsing.parsers.constraintparser.states.variable import VariableStateDescription
from services.nlg.templates.parsing.parsers.constraintparser.states.variableend import VariableEndStateDescription


class ConstraintParser(ModifiedPushdownAutomaton):
    def __init__(self):
        ModifiedPushdownAutomaton.__init__(self, ConstraintStartState(), [ConstraintEndState()], [
            ConstraintStartStateDescription(),
            VariableStateDescription(),
            VariableEndStateDescription(),
            OperatorStateDescription(),
            OperatorEndStateDescription(),
            ValueStateDescription(),
            ValueEscapeStateDescription(),
            ConstraintEndStateDescription()
        ])


if __name__ == '__main__':
    automaton = ConstraintParser()
    # automaton.parse('ab(cd.ef)$')
    automaton.parse('a="b"&a="c"')
    print(automaton.stack.data_stack)
