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

"""Handcrafted (i.e. template-based) Natural Language Generation Module with backchannel"""

import inspect
import os

from services.service import PublishSubscribe
from utils.common import Language
from utils.domain.domain import Domain
from utils.logger import DiasysLogger
from utils.sysact import SysAct
from services.nlg import HandcraftedNLG
from typing import Dict


class BackchannelHandcraftedNLG(HandcraftedNLG):
    """Handcrafted (i.e. template-based) Natural Language Generation Module

    A rule-based approach on natural language generation.
    The rules have to be specified within a template file using the ADVISER NLG syntax.
    Python methods that are called within a template file must be specified in the
    HandcraftedNLG class by using the prefix "_template_". For example, the method
    "_template_genitive_s" can be accessed in the template file via calling {genitive_s(name)}

    Attributes:
        domain (Domain): the domain
        template_filename (str): the NLG template filename
        templates (TemplateFile): the parsed and ready-to-go NLG template file
        template_english (str): the name of the English NLG template file
        template_german (str): the name of the German NLG template file
        language (Language): the language of the dialogue
    """
    def __init__(self, domain: Domain, sub_topic_domains: Dict[str, str] = {}, template_file: str = None,
                 logger: DiasysLogger = DiasysLogger(), template_file_german: str = None,
                 language: Language = None):
        """Constructor mainly extracts methods and rules from the template file"""
        HandcraftedNLG.__init__(
            self, domain, template_file=None,
            logger=DiasysLogger(), template_file_german=None,
            language=None, sub_topic_domains=sub_topic_domains)

        # class_int_mapping = {0: b'no_bc', 1: b'assessment', 2: b'continuer'}
        self.backchannels = {
            0: [''],
            1: ['Okay. ', 'Yeah. '],
            2: ['Um-hum. ', 'Uh-huh. ']
        }

    @PublishSubscribe(sub_topics=["sys_act", 'predicted_BC'], pub_topics=["sys_utterance"])
    def publish_system_utterance(self, sys_act: SysAct = None, predicted_BC: int = None) -> dict(sys_utterance=str):
        """
        Takes a system act, searches for a fitting rule, adds, backchannel and applies it
        and returns the message.
        mapping = {0: b'no_bc', 1: b'assessment', 2: b'continuer'}

        Args:
            sys_act (SysAct): The system act, to check whether the dialogue was finished
            predicted_BC (int): integer representation of the BC

        Returns:
            dict: a dict containing the system utterance
        """
        rule_found = True
        message = self.generate_system_utterance(sys_act)

        if 'Sorry' not in message:
            message = self.backchannels[predicted_BC][0] + message

        return {'sys_utterance': message}
