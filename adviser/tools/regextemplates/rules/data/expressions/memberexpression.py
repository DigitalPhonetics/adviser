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

from tools.regextemplates.rules.data.expressions.expression import Expression
from tools.regextemplates.rules.data.memory import Memory

class MemberExpression(Expression):
    def __init__(self, variable_name: str, member_attribute: str):
        Expression.__init__(self)
        self.variable = variable_name
        self.attribute = member_attribute

    def evaluate(self, environment: Memory) -> str:
        variable_value = environment.get_variable_value(self.variable)
        return environment.get_member(variable_value, self.attribute)

    def __repr__(self):
        return f'{self.variable}->{self.attribute}'
