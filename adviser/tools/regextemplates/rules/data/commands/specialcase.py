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

from typing import List, Tuple
from random import choices

from tools.regextemplates.rules.data.commands.command import Command
from tools.regextemplates.rules.data.commands.message import Message
from tools.regextemplates.rules.data.memory import Memory
from tools.regextemplates.rules.data.expressions.expression import Expression
from tools.regextemplates.rules.parsing.parsers.constraintparser.constraintparser import ConstraintParser


CONSTRAINT_PARSER = ConstraintParser()


class SpecialCase(Command):
    def __init__(self, arguments: str):
        Command.__init__(self, arguments)
        self.constraints = self._parse_arguments()

        self.messages = []
        self.special_cases = []
        self.additions = []

    def _parse_arguments(self) -> List[Tuple[Expression, Expression]]:
        constraints = []
        parsed_constraints = CONSTRAINT_PARSER.parse(self.arguments)
        for parsed_constraint in parsed_constraints:
            constraints.append((parsed_constraint.left_side, parsed_constraint.right_side))
        return constraints

    def are_arguments_valid(self) -> bool:
        return True  # since parse_arguments would have thrown an exception otherwise

    def add_inner_command(self, command):
        if isinstance(command, Message):
            if self._were_special_cases_added_before():
                raise ValueError('Messages must be added to a special case BEFORE special cases')
            self.messages.append(command)
        elif isinstance(command, SpecialCaseException):
            self.special_cases.append(command)
        elif isinstance(command, SpecialCaseAddition):
            self.additions.append(command)
        else:
            raise ValueError('Expected "message" or "special_case" commands only!')

    def _were_special_cases_added_before(self) -> bool:
        return len(self.special_cases) > 0

    def are_inner_commands_valid(self) -> bool:
        return len(self.messages) + len(self.special_cases) > 0

    def is_applicable(self, parameters: Memory) -> bool:
        for constraint in self.constraints:
            if constraint[0].evaluate(parameters) != constraint[1].evaluate(parameters):
                return False
        return True

    def apply(self, parameters: Memory) -> str:
        special_case = self._get_applicable_special_case(parameters)
        if special_case is not None:
            return special_case.apply(parameters)

        all_regexes = []
        for message in self._extract_potential_messages(parameters):
            all_regexes.append(message.apply(parameters))
        return '(' + '|'.join(all_regexes) + ')'

    def _get_applicable_special_case(self, parameters: Memory):
        for special_case in self.special_cases:
            if special_case.is_applicable(parameters):
                return special_case
        return None
    
    def _extract_potential_messages(self, parameters):
        messages = self.messages[:]
        for addition in self.additions:
            if addition.is_applicable(parameters):
                messages.extend(addition.get_additional_messages(parameters))
        return messages

    def _select_message(self, messages: List[Message]):
        if not messages:
            raise ValueError('No constraints were matched and no default message was specified!')
        if len(messages) > 1:
            scores = [message.score for message in messages]
            return choices(messages, scores)[0]  # choices returns a list of length 1
        return messages[0]


class SpecialCaseException(SpecialCase):
    def __init__(self, arguments: str):
        SpecialCase.__init__(self, arguments)


class SpecialCaseAddition(SpecialCase):
    def __init__(self, arguments: str):
        SpecialCase.__init__(self, arguments)
    
    def get_additional_messages(self, parameters: Memory) -> List[Message]:
        special_case = self._get_applicable_special_case(parameters)
        if special_case is not None:
            return special_case._extract_potential_messages(parameters)
        return self._extract_potential_messages(parameters)
