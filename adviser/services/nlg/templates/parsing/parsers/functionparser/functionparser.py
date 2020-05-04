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

from services.nlg.templates.parsing.parsers.functionparser.states.statelist import FunctionNameState, AcceptState

from services.nlg.templates.parsing.parsers.functionparser.states.accept import AcceptStateDescription
from services.nlg.templates.parsing.parsers.functionparser.states.argument import ArgumentStateDescription
from services.nlg.templates.parsing.parsers.functionparser.states.argumentstart import ArgumentStartStateDescription
from services.nlg.templates.parsing.parsers.functionparser.states.freeargument import FreeArgumentStateDescription
from services.nlg.templates.parsing.parsers.functionparser.states.functionname import FunctionNameStateDescription
from services.nlg.templates.parsing.parsers.functionparser.states.requiredargument import \
    RequiredArgumentStateDescription


class FunctionParser(ModifiedPushdownAutomaton):
    def __init__(self):
        ModifiedPushdownAutomaton.__init__(self, FunctionNameState(), [AcceptState()], [
            FunctionNameStateDescription(),
            ArgumentStartStateDescription(),
            ArgumentStateDescription(),
            FreeArgumentStateDescription(),
            RequiredArgumentStateDescription(),
            AcceptStateDescription()
        ])

if __name__ == '__main__':
    automaton = FunctionParser()
    automaton.parse('intent(a,*b)')
    print([str(elem) for elem in automaton.stack.data_stack])
