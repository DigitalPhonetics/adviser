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

from services.nlg.templates.parsing.configuration import \
    StateDescription, Configuration, DefaultTransition, TransitionWithoutAction
from services.nlg.templates.parsing.parsers.messageparser.states.statelist import \
    CodeState, AdviserState, PythonState
from services.nlg.templates.parsing.stack import AutomatonStack


class _CodeDefaultTransition(DefaultTransition):
    """This is called whenever an opening brace was not followed by another opening brace"""
    def __init__(self):
        DefaultTransition.__init__(self, CodeState())

    def get_output_configuration(self, input_configuration: Configuration) -> Configuration:
        return Configuration(AdviserState(), input_configuration.character)

    def perform_stack_action(self, stack: AutomatonStack, configuration: Configuration):
        pass


class CodeStateDescription(StateDescription):
    def __init__(self):
        StateDescription.__init__(self, CodeState(), _CodeDefaultTransition(), [
            TransitionWithoutAction(Configuration(CodeState(), '{'),
                                    Configuration(PythonState(), ''))
        ])
