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

from typing import Tuple, List, Dict, Callable
import sys

from utils.useract import UserAct
from utils.domain.jsonlookupdomain import JSONLookupDomain

from tools.regextemplates.rules.data.memory import Memory, Variable, GlobalMemory
from tools.regextemplates.rules.data.commands.command import Command
from tools.regextemplates.rules.data.commands.rule import Rule
from tools.regextemplates.rules.data.commands.function import Function
from tools.regextemplates.rules.data.commands.specialcase import SpecialCaseException, SpecialCaseAddition
from tools.regextemplates.rules.data.commands.message import Message
from tools.regextemplates.rules.data.commands.probability import Probability

from tools.regextemplates.rules.builtinfunctions import PythonFunction, ForFunction, ForEntryFunction, ForEntryListFunction
from tools.regextemplates.rules.preprocessing import _Preprocessor


KEYWORDS = {
    'rule': Rule,
    'useract': Rule,
    'function': Function,
    'method': Function,
    'special_case': SpecialCaseException,
    'exception': SpecialCaseException,
    'if': SpecialCaseException,
    'addition': SpecialCaseAddition,
    'add': SpecialCaseAddition,
    'add_if': SpecialCaseAddition,
    'on_top': SpecialCaseAddition,
    'regex': Message,
    'message': Message,
    'text': Message,
    'prob': Probability,
    'probability': Probability,
    'with': Probability,
    'weight': Probability
}


class RegexFile:
    """Interprets a regex file
    
    Attributes:
        global_memory {GlobalMemory} -- memory that can be accessed at all times in the tempaltes
    """

    def __init__(self, filename: str, domain: JSONLookupDomain):
        self.global_memory = GlobalMemory(domain)
        self._add_built_in_functions()
        tfr = _RuleFileReader(filename)
        self._rules = self._create_rule_dict(tfr.get_rules())
        self._add_functions_to_global_memory(tfr.get_functions())

    def _add_built_in_functions(self):
        self.global_memory.add_function(ForFunction(self.global_memory))
        self.global_memory.add_function(ForEntryFunction(self.global_memory))
        self.global_memory.add_function(ForEntryListFunction(self.global_memory))
    
    def _create_rule_dict(self, rules: List[Rule]) -> Dict[str, Rule]:
        rule_dict = {}
        for rule in rules:
            if rule.intent not in rule_dict:
                rule_dict[rule.intent] = []
            rule_dict[rule.intent].append(rule)
        return rule_dict
    
    def _add_functions_to_global_memory(self, functions: List[Function]):
        for function in functions:
            self.global_memory.add_function(function)

    def create_regex(self, user_act: UserAct) -> str:
        """Iterates through all possible rules and applies the first one to fit the user act
        
        Arguments:
            user_act {UserAct} -- the system act to find a rule for
        
        Raises:
            BaseException: when no rule could be applied
        
        Returns:
            str -- the message returned by the rule
        """
        slots = self._create_memory_from_user_act(user_act)
        for rule in self._rules[user_act.type.value]:
            if rule.is_applicable(slots):
                return rule.apply(slots)
        raise BaseException('No rule was found for the given system act.')

    def _create_memory_from_user_act(self, user_act: UserAct) -> Memory:
        slots = Memory(self.global_memory)
        if user_act.slot is not None:
            slots.add_variable(Variable(user_act.slot, user_act.value))
        return slots

    def add_python_function(self, function_name: str, python_function: Callable[[object], str],
                            obligatory_arguments: List[object] = []):
        """Add a python function to the global memory of the rule file interpreter
        
        Arguments:
            function_name {str} -- name under which the function can be accessed in rule file
            python_function {Callable[[object], str]} -- python function which is called when being
                accessed in the rule file
        
        Keyword Arguments:
            obligatory_arguments {List[object]} -- objects that are always passed as first
                arguments to the python function, e.g. "self" (default: {[]})
        """
        self.global_memory.add_function(PythonFunction(function_name, python_function,
                                                       obligatory_arguments))


class _RuleFileReader:
    def __init__(self, filename: str):
        self._filename = filename
        self._rules: List[Command] = []
        self._functions: List[Function] = []
        self._command_stack: List[Command] = []
        self._current_line: str = ''
        self._current_block_level: int = 0
        self._current_command: Command = None

        self._content = ''
        self._load_file()
    
    def get_rules(self):
        return self._rules
    
    def get_functions(self):
        return self._functions

    def _load_file(self):
        preprocessor = _Preprocessor(self._filename)
        lines = preprocessor.get_preprocessed_lines()
        for line_no, line in enumerate(lines):
            self._current_line = line
            try:
                self._process_line()
            except BaseException as error:
                print(f'Error in line {line_no+1}: "{line}"')
                print(error)
                sys.exit(1)

        self._current_block_level = 0
        self._pop_commands_from_stack()

    def _process_line(self):
        self._current_block_level = self._get_block_level()
        keyword, arguments = self._get_keyword_and_arguments()
        self._current_command = KEYWORDS[keyword](arguments)

        self._check_command_arguments()
        self._check_block_level()
        self._pop_commands_from_stack()
        self._add_new_command()

    def _get_block_level(self) -> int:
        whitespace_count = 0
        for character in self._current_line:
            if character != ' ':
                break
            whitespace_count += 1
        if whitespace_count % 4 != 0:
            raise BaseException('Block indentation failed! Block indent must contain 4 spaces!')
        return whitespace_count // 4

    def _get_keyword_and_arguments(self) -> Tuple[str, str]:
        parts = self._current_line.strip().split(maxsplit=1)
        if not parts or parts[0] not in KEYWORDS:
            raise BaseException('No keyword detected in line "{self._current_line}"')
        return parts[0], parts[1]

    def _check_command_arguments(self):
        if not self._current_command.are_arguments_valid():
            raise BaseException('Command arguments invalid!')

    def _check_block_level(self):
        max_block_level = len(self._command_stack) + 1
        if self._current_block_level > max_block_level:
            raise BaseException(f'Block indent to high (got {self._current_block_level}, '
                                f'expected max. {max_block_level})')

    def _pop_commands_from_stack(self):
        while len(self._command_stack) > self._current_block_level:
            command = self._command_stack.pop()
            if not command.are_inner_commands_valid():
                raise BaseException(f'Command block has been closed without command being valid!')

    def _add_new_command(self):
        if self._current_block_level == 0:
            self._add_top_level_command()
        else:
            self._command_stack[-1].add_inner_command(self._current_command)
        self._command_stack.append(self._current_command)

    def _add_top_level_command(self):
        if isinstance(self._current_command, Rule):
            self._rules.append(self._current_command)
        elif isinstance(self._current_command, Function):
            self._functions.append(self._current_command)
        else:
            raise BaseException('Only function or rule commands can be defined on top level!')

