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

from typing import List, Callable

from services.nlg.templates.data.commands.function import Function
from services.nlg.templates.data.memory import Memory, Variable, GlobalMemory


class PythonFunction(Function):
    def __init__(self, function_name: str, function_to_call: Callable,
                 obligatory_arguments: List[object] = []):
        Function.__init__(self, f'{function_name}()')
        self.function = function_to_call
        self.obligatory_arguments = obligatory_arguments

    def is_applicable(self, parameters: Memory) -> bool:
        return True

    def apply(self, parameters: Memory = None) -> str:
        arguments = self.obligatory_arguments.copy()
        arguments.extend([variable.value for variable in parameters.variables])
        return self.function(*arguments)


class ForEntryFunction(Function):
    def __init__(self, global_memory):
        Function.__init__(self, 'for_entry(slots, function, separator_first, separator_last)')
        self.global_memory = global_memory

    def is_applicable(self, parameters: Memory) -> bool:
        return len(parameters.variables) >= 4

    def apply(self, parameters: Memory = None) -> str:
        function = parameters.get_function(parameters.variables[1].value)
        extra_arguments = [variable.value for variable in parameters.variables[4:]]
        texts: List[str] = []

        for slot_value_pair in parameters.variables[0].value:
            memory = self._build_memory(slot_value_pair[0], slot_value_pair[1], extra_arguments)
            if not function.is_applicable(memory):
                raise BaseException(f'The function {function.function_name} could not be called '
                                    f'from the for_entry function')
            texts.append(function.apply(memory))

        return self._create_text_from_elements(texts, parameters.variables[2].value,
                                               parameters.variables[3].value)

    def _build_memory(self, slot: str, value: str, arguments: List[str]):
        memory = Memory(self.global_memory)
        memory.add_variable(Variable('slot', slot))
        memory.add_variable(Variable('value', value))
        for i, argument in enumerate(arguments):
            memory.add_variable(Variable(f'arg{i}', argument))
        return memory

    def _create_text_from_elements(self, elements: List[str], separator: str, last_separator: str):
        if not elements:
            return ''
        if len(elements) == 1:
            return elements[0]
        text = elements[0]
        for i in range(1, len(elements) - 1):
            text += separator + elements[i]
        text += last_separator + elements[-1]
        return text


class ForFunction(Function):
    def __init__(self, global_memory):
        Function.__init__(self, 'for(values, function, separator_first, separator_last, *args)')
        self.global_memory = global_memory

    def is_applicable(self, parameters: Memory) -> bool:
        return len(parameters.variables) >= 4

    def apply(self, parameters: Memory = None) -> str:
        function = parameters.get_function(parameters.variables[1].value)
        extra_arguments = [variable.value for variable in parameters.variables[4:]]
        texts: List[str] = []

        for value in parameters.variables[0].value:
            memory = self._build_memory(value, extra_arguments)
            if not function.is_applicable(memory):
                raise BaseException(f'The function {function.function_name} could not be called '
                                    f'from the for function')
            texts.append(function.apply(memory))

        return self._create_text_from_elements(texts, parameters.variables[2].value,
                                               parameters.variables[3].value)

    def _build_memory(self, value: str, arguments: List[str]):
        memory = Memory(self.global_memory)
        memory.add_variable(Variable('value', value))
        for i, argument in enumerate(arguments):
            memory.add_variable(Variable(f'arg{i}', argument))
        return memory

    def _create_text_from_elements(self, elements: List[str], separator: str, last_separator: str):
        if not elements:
            return ''
        if len(elements) == 1:
            return elements[0]
        text = elements[0]
        for i in range(1, len(elements) - 1):
            text += separator + elements[i]
        text += last_separator + elements[-1]
        return text


class ForEntryListFunction(Function):
    def __init__(self, global_memory: GlobalMemory):
        Function.__init__(self, 'for_entry_list(slots, function, value_sep, value_sep_last, '
                                'slot_sep, slot_sep_last)')
        self.global_memory = global_memory

    def is_applicable(self, parameters: Memory) -> bool:
        return len(parameters.variables) >= 6

    def apply(self, parameters: Memory = None) -> str:
        function = parameters.get_function(parameters.variables[1].value)
        extra_arguments = [variable.value for variable in parameters.variables[6:]]
        texts_per_slot: List[str] = []

        for slot_values_pair in parameters.variables[0].value:
            slot_texts: List[str] = []
            for value in slot_values_pair[1]:            
                memory = self._build_memory(slot_values_pair[0], value, extra_arguments)
                if not function.is_applicable(memory):
                    raise BaseException(f'The function {function.function_name} could not be '
                                        f'called from the for_entry_list function')
                slot_texts.append(function.apply(memory))
            text = self._create_text_from_elements(slot_texts, parameters.variables[2].value,
                                                   parameters.variables[3].value)
            texts_per_slot.append(text)

        return self._create_text_from_elements(texts_per_slot, parameters.variables[4].value,
                                               parameters.variables[5].value)

    def _build_memory(self, slot: str, value: str, arguments: List[str]):
        memory = Memory(self.global_memory)
        memory.add_variable(Variable('slot', slot))
        memory.add_variable(Variable('value', value))
        for i, argument in enumerate(arguments):
            memory.add_variable(Variable(f'arg{i}', argument))
        return memory

    def _create_text_from_elements(self, elements: List[str], separator: str, last_separator: str):
        if not elements:
            return ''
        if len(elements) == 1:
            return elements[0]
        text = elements[0]
        for i in range(1, len(elements) - 1):
            text += separator + elements[i]
        text += last_separator + elements[-1]
        return text
