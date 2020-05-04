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

from services.nlg.templates.parsing.parsers.messageparser.states.statelist import StartState, AcceptState

from services.nlg.templates.parsing.parsers.messageparser.states.accept import AcceptStateDescription
from services.nlg.templates.parsing.parsers.messageparser.states.adviser import AdviserStateDescription
from services.nlg.templates.parsing.parsers.messageparser.states.code import CodeStateDescription
from services.nlg.templates.parsing.parsers.messageparser.states.codestring import CodeStringStateDescription
from services.nlg.templates.parsing.parsers.messageparser.states.escape import EscapeStateDescription
from services.nlg.templates.parsing.parsers.messageparser.states.message import MessageStateDescription
from services.nlg.templates.parsing.parsers.messageparser.states.python import PythonStateDescription
from services.nlg.templates.parsing.parsers.messageparser.states.pythonclosingbrace import \
    PythonClosingBraceStateDescription
from services.nlg.templates.parsing.parsers.messageparser.states.start import StartStateDescription


class MessageParser(ModifiedPushdownAutomaton):
    def __init__(self):
        ModifiedPushdownAutomaton.__init__(self, StartState(), [AcceptState()], [
            StartStateDescription(),
            AcceptStateDescription(),
            MessageStateDescription(),
            EscapeStateDescription(),
            CodeStateDescription(),
            AdviserStateDescription(),
            PythonStateDescription(),
            PythonClosingBraceStateDescription(),
            CodeStringStateDescription()
        ])


if __name__ == '__main__':
    automaton = MessageParser()
    automaton.parse('"{pers_pron(name.gender)} has the number {phone} \\"and\\" {{genitive_s(name.short_name, "\\"f\\n")}} office in {room}."')
    print(automaton.stack.data_stack)
