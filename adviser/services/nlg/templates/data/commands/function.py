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

from random import choices
from typing import List

from services.nlg.templates.data.commands.command import Command
from services.nlg.templates.data.commands.message import Message
from services.nlg.templates.data.commands.specialcase import SpecialCaseException, SpecialCaseAddition
from services.nlg.templates.data.memory import Memory, Variable, GlobalMemory
from services.nlg.templates.parsing.parsers.functionparser.functionparser import FunctionParser

FUNCTION_PARSER = FunctionParser()


class Function(Command):
    def __init__(self, arguments: str):
        Command.__init__(self, arguments)
        self.function_name = None
        self.argument_names = []
        self.free_parameter = None
        self._parse_arguments()

        self.messages = []
        self.special_cases = []
        self.additions = []

    def _parse_arguments(self):
        parsed_objects = FUNCTION_PARSER.parse(self.arguments.replace(' ', ''))
        self.function_name = parsed_objects[0].function_name
        for i in range(1, len(parsed_objects)):
            self.argument_names.append(parsed_objects[i].variable_name)
        if len(parsed_objects) > 2 and parsed_objects[-1].free_variable:
            self.free_parameter = self.argument_names.pop()

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
        if self.free_parameter is not None:
            return len(parameters.variables) >= len(self.argument_names)
        return len(parameters.variables) == len(self.argument_names)

    def apply(self, parameters: Memory = None) -> str:
        argument_values = [variable.value for variable in parameters.variables]
        if self.free_parameter is not None:
            variables = self._build_memory_with_free_parameter(argument_values,
                                                               parameters.global_memory)
        else:
            variables = self._build_memory_without_free_parameter(argument_values,
                                                                  parameters.global_memory)

        special_case = self._get_applicable_special_case(variables)
        if special_case is not None:
            return special_case
        potential_messages = self._extract_potential_messages(variables)
        return self._select_message(potential_messages).apply(variables)

    def _build_memory_without_free_parameter(self, argument_values: List[object],
                                             global_memory: GlobalMemory) -> Memory:
        assert len(self.argument_names) == len(argument_values)
        variables = Memory(global_memory)
        for i in range(len(self.argument_names)):
            variables.add_variable(Variable(self.argument_names[i], argument_values[i]))
        return variables

    def _build_memory_with_free_parameter(self, argument_values: List[object]) -> Memory:
        variables = Memory(global_memory)
        for i in range(len(self.argument_names)):
            variables.add_variable(Variable(self.argument_names[i], argument_values[i]))

        variables.add_variable(Variable(self.free_parameter.variable_name,
                                        argument_values[len(self.argument_names):]))
        return variables

    def _get_applicable_special_case(self, parameters: Memory):
        for special_case in self.special_cases:
            if special_case.is_applicable(parameters):
                return special_case.apply(parameters)
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
