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

from services.nlg.templates.parsing.parsers.codeparser.states.statelist import ExpressionState, AcceptState

from services.nlg.templates.parsing.parsers.codeparser.states.accept import AcceptStateDescription
from services.nlg.templates.parsing.parsers.codeparser.states.expression import ExpressionStateDescription
from services.nlg.templates.parsing.parsers.codeparser.states.expressionend import ExpressionEndStateDescription
from services.nlg.templates.parsing.parsers.codeparser.states.member import MemberStateDescription
from services.nlg.templates.parsing.parsers.codeparser.states.optionalexpression import \
    OptionalExpressionStateDescription
from services.nlg.templates.parsing.parsers.codeparser.states.string import StringStateDescription
from services.nlg.templates.parsing.parsers.codeparser.states.stringescape import StringEscapeStateDescription
from services.nlg.templates.parsing.parsers.codeparser.states.variable import VariableStateDescription


class CodeParser(ModifiedPushdownAutomaton):
    def __init__(self):
        ModifiedPushdownAutomaton.__init__(self, ExpressionState(), [AcceptState()], [
            ExpressionStateDescription(),
            OptionalExpressionStateDescription(),
            AcceptStateDescription(),
            StringStateDescription(),
            StringEscapeStateDescription(),
            VariableStateDescription(),
            MemberStateDescription(),
            ExpressionEndStateDescription()
        ])


if __name__ == '__main__':
    automaton = CodeParser()
    # automaton.parse('ab(cd.ef)$')
    automaton.parse('func(name.gender, test("abc"))$')
    print(automaton.stack.data_stack)
