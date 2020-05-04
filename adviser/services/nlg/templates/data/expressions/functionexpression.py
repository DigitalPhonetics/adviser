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

from services.nlg.templates.data.expressions.expression import Expression
from services.nlg.templates.data.memory import Memory, Variable


class FunctionExpression(Expression):
    def __init__(self, name: str, arguments: List[Expression]) -> str:
        Expression.__init__(self)
        self.name = name
        self.arguments = arguments

    def evaluate(self, environment: Memory) -> str:
        function = environment.get_function(self.name)
        parameters = Memory(environment.global_memory)
        for i in range(len(self.arguments)):
            parameters.add_variable(Variable(f'arg{i}', self.arguments[i].evaluate(environment)))
        return function.apply(parameters)

    def __repr__(self):
        argument_string = ', '.join([repr(argument) for argument in self.arguments])
        return f'{self.name}({argument_string})'
