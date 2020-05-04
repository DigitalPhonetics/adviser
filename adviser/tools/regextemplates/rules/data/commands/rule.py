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

from typing import List, Dict
from random import choices

from tools.regextemplates.rules.data.commands.command import Command
from tools.regextemplates.rules.data.commands.message import Message
from tools.regextemplates.rules.data.commands.specialcase import SpecialCaseException, SpecialCaseAddition
from tools.regextemplates.rules.data.memory import Memory, Variable, GlobalMemory
from tools.regextemplates.rules.parsing.parsers.functionparser.functionparser import FunctionParser


FUNCTION_PARSER = FunctionParser()


class Rule(Command):
    def __init__(self, arguments: str):
        Command.__init__(self, arguments)
        self.intent = None
        self.slot_names = []
        self.free_parameter = None
        self._parse_arguments()

        self.messages = []
        self.special_cases = []
        self.additions = []

    def _parse_arguments(self):
        parsed_objects = FUNCTION_PARSER.parse(self.arguments.replace(' ', ''))
        self.intent = parsed_objects[0].function_name
        for i in range(1, len(parsed_objects)):
            self.slot_names.append(parsed_objects[i].variable_name)
        if len(parsed_objects) > 1 and parsed_objects[-1].free_variable:
            self.free_parameter = self.slot_names.pop()

    def are_arguments_valid(self) -> bool:
        return True  # since parse_arguments would have thrown an exception otherwise

    def add_inner_command(self, command):
        if isinstance(command, Message):
            if self._were_special_cases_added_before():
                raise ValueError('Messages must be added to the function BEFORE special cases')
            self.messages.append(command)
        elif isinstance(command, SpecialCaseException):
            self.special_cases.append(command)
        elif isinstance(command, SpecialCaseAddition):
            self.additions.append(command)
        else:
            raise ValueError('Expected Message or SpecialCase commands only!')

    def _were_special_cases_added_before(self) -> bool:
        return len(self.special_cases) > 0

    def are_inner_commands_valid(self) -> bool:
        return len(self.messages) + len(self.special_cases) > 0

    def is_applicable(self, parameters: Memory) -> bool:
        slot_names_to_check = [slot.name for slot in parameters.variables]

        for template_slot in self.slot_names:
            if template_slot not in slot_names_to_check:
                return False
            slot_names_to_check.remove(template_slot)

        return len(slot_names_to_check) == 0 or self.free_parameter is not None

    def apply(self, parameters: Memory = None) -> str:
        slot_dict = parameters.variable_dict.copy()
        if self.free_parameter is not None:
            variables = self._build_memory_with_free_parameter(slot_dict,
                                                               parameters.global_memory)
        else:
            variables = self._build_memory_without_free_parameter(slot_dict,
                                                                  parameters.global_memory)

        special_case = self._get_applicable_special_case(variables)
        if special_case is not None:
            return special_case.apply(variables)
        
        all_regexes = []
        for message in self._extract_potential_messages(variables):
            all_regexes.append(message.apply(variables))
        return '(' + '|'.join(all_regexes) + ')'

    def _build_memory_without_free_parameter(self, slot_dict: Dict[str, object],
                                             global_memory: GlobalMemory) -> Memory:
        assert len(self.slot_names) == len(slot_dict)
        variables = Memory(global_memory)
        for slot in self.slot_names:
            value = self._flatten_value_list(slot_dict[slot])
            variables.add_variable(Variable(slot, value))
        return variables

    def _build_memory_with_free_parameter(self, slot_dict: Dict[str, object],
                                          global_memory: GlobalMemory) -> Memory:
        variables = Memory(global_memory)
        for slot in self.slot_names:
            value = self._flatten_value_list(slot_dict[slot])
            variables.add_variable(Variable(slot, value))
            slot_dict.pop(slot)

        slot_value_pairs = []
        for slot in slot_dict:
            value = self._flatten_value_list(slot_dict[slot])
            slot_value_pairs.append((slot, value))
        variables.add_variable(Variable(self.free_parameter, slot_value_pairs))

        return variables

    def _flatten_value_list(self, value_list: List[object]):
        if not value_list:
            return None
        if len(value_list) == 1:
            return value_list[0]
        return value_list

    def _get_applicable_special_case(self, parameters: Memory) -> SpecialCaseException:
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
