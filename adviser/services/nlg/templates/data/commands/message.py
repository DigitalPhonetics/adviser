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

from typing import List

from services.nlg.templates.data.commands.command import Command
from services.nlg.templates.data.commands.probability import Probability
from services.nlg.templates.data.memory import Memory
from services.nlg.templates.parsing.parsers.codeparser.codeparser import CodeParser
from services.nlg.templates.parsing.parsers.messageparser.data.messagecomponent import MessageComponent, \
    MessageComponentType
from services.nlg.templates.parsing.parsers.messageparser.messageparser import MessageParser

MESSAGE_PARSER = MessageParser()
CODE_PARSER = CodeParser()


class Message(Command):
    def __init__(self, arguments: str):
        Command.__init__(self, arguments)

        self.components = self._parse_message()
        self.score = 1.0

    def _parse_message(self) -> List[MessageComponent]:
        return MESSAGE_PARSER.parse(self.arguments)

    def are_arguments_valid(self) -> bool:
        return True  # since parse_arguments would have thrown an exception otherwise

    def add_inner_command(self, command):
        assert isinstance(command, Probability)
        self.score = command.value
    
    def get_normalised_probability(self, score_sum: float) -> float:
        return self.score / score_sum

    def are_inner_commands_valid(self) -> bool:
        return len(self.inner_commands) <= 1

    def is_applicable(self, parameters: Memory) -> bool:
        return True  # messages are always applicable

    def apply(self, parameters: Memory) -> str:
        output = ''
        for component in self.components:
            if component.component_type == MessageComponentType.STRING:
                output += component.value
            elif component.component_type == MessageComponentType.ADVISER_CODE or \
                component.component_type == MessageComponentType.PYTHON_CODE:
                code = component.value + '$'  # CodeParser expects end of statement
                expression = CODE_PARSER.parse(code)[0]
                output += expression.evaluate(parameters)
        return output
