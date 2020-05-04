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
from services.nlg.templates.data.expressions.functionexpression import FunctionExpression


class FunctionDeclaration:
    def __init__(self, function_name: str):
        self.function_name = function_name

    def create_function(self, arguments: List[Expression]) -> FunctionExpression:
        return FunctionExpression(self.function_name, arguments)
