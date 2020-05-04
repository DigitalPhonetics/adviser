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

from typing import Dict, List, Callable
from utils.domain.jsonlookupdomain import JSONLookupDomain


class _FunctionDescription:
    def __init__(self, name: str, number_of_arguments: int):
        self.name = name
        self.argument_count = number_of_arguments

    def __hash__(self):
        return hash(self.name) * hash(self.argument_count)


class Variable:
    def __init__(self, variable_name, value):
        self.name = variable_name
        self.value = value


class Memory:
    def __init__(self, global_memory: 'GlobalMemory'):
        self.variables: List[Variable] = []
        self.variable_dict: Dict[str, object] = {}
        self.functions: List['Function'] = []
        self.function_dict: Dict[str, 'Function'] = {}
        self.global_memory = global_memory

    def add_variable(self, variable: Variable):
        self.variables.append(variable)
        self.variable_dict[variable.name] = variable.value

    def get_variable_value(self, variable_name: str) -> object:
        if variable_name not in self.variable_dict.keys():
            if (not self.global_memory) or (variable_name not in self.global_memory.variable_dict):
                raise ValueError(f'No such variable {variable_name} found.')
            return self.global_memory.variable_dict[variable_name]
        return self.variable_dict[variable_name]

    def add_function(self, function: 'Function'):
        self.functions.append(function)
        self.function_dict[function.function_name] = function

    def get_function(self, function_name: str) -> 'Function':
        if function_name not in self.function_dict:
            if not self.global_memory or function_name not in self.global_memory.function_dict:
                raise ValueError(f'No such function "{function_name}" found.')
            return self.global_memory.function_dict[function_name]
        return self.function_dict[function_name]
    
    def get_member(self, primary_key_value: str, attribute_name: str) -> str:
        assert self.global_memory is not None
        return self.global_memory.get_member(primary_key_value, attribute_name)


class GlobalMemory(Memory):
    def __init__(self, domain: JSONLookupDomain):
        Memory.__init__(self, None)
        self.domain = domain
    
    def get_member(self, primary_key_value: str, attribute_name: str) -> str:
        primary_key_name = self.domain.get_primary_key()
        table_name = self.domain.get_domain_name()
        query_result = self.domain.query_db(
            f'SELECT {attribute_name} FROM {table_name} '
            f'WHERE {primary_key_name} = \'{primary_key_value}\'')
        if not query_result:
            raise ValueError(f"Couldn't find an entry for primary key {primary_key_value}.")
        return query_result[0][attribute_name]
