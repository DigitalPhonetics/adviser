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

"""Handcrafted (i.e. template-based) Natural Language Generation Module"""

import os
import inspect

from services.nlg import HandcraftedNLG
from services.nlg.templates.templatefile import TemplateFile
from utils.logger import DiasysLogger
from utils.sysact import SysAct
from utils.domain.domain import Domain
from utils.common import Language
from services.service import Service
from services.service import PublishSubscribe
from typing import List


class HandcraftedEmotionNLG(HandcraftedNLG):
    """
        A child of the HandcraftedNLG, the HandcraftedEmotionNLG can choose between multiple affective
        response templates for each sys_act dependingon the current sys_emotion
    """
    def __init__(self, domain: Domain, sub_topic_domains={}, template_file: str = None,
                 logger: DiasysLogger = DiasysLogger(), template_file_german: str = None,
                 emotions: List[str] = [], debug_logger = None):
        """Constructor mainly extracts methods and rules from the template file"""
        Service.__init__(self, domain=domain, sub_topic_domains=sub_topic_domains, debug_logger=debug_logger)

        self.domain = domain
        self.template_filename = template_file
        self.templates = {}
        self.logger = logger
        self.emotions = emotions

        self._initialise_templates()


    @PublishSubscribe(sub_topics=["sys_act", "sys_emotion", "sys_engagement"], pub_topics=["sys_utterance"])
    def generate_system_utterance(self, sys_act: SysAct = None, sys_emotion: str = None,
                                  sys_engagement: str = None) -> dict(sys_utterance=str):
        """

        Takes a system act, system emotion choice, and system engagement level choice, then
        searches for a fitting rule, applies it and returns the message.

        Args:
            sys_act (SysAct): The system act, to check whether the dialogue was finished
            sys_emotion (str): A string representing the system's choice of emotional response
            sys_engagement (str): A string representing how engaged the system thinks the user is
 
        Returns:
            dict: a dict containing the system utterance
        """
        rule_found = True
        message = ""
        try:
            message = self.templates[sys_emotion].create_message(sys_act)
        except BaseException as error:
            rule_found = False
            self.logger.error(error)
            raise(error)

        # inform if no applicable rule could be found in the template file
        if not rule_found:
            self.logger.info('Could not find a fitting rule for the given system act!')
            self.logger.info("System Action: " + str(sys_act.type)
                             + " - Slots: " + str(sys_act.slot_values))

        # self.logger.dialog_turn("System Action: " + message)
        return {'sys_utterance': message}

    def _initialise_templates(self):
        """
            Loads the correct template file based on which language has been selected
            this should only be called on the first turn of the dialog

            Args:
                language (Language): Enum representing the language the user has selected
        """
        for emotion in self.emotions:
            self.templates[emotion.lower()] = TemplateFile(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f'../../resources/nlg_templates/{self.domain.get_domain_name()}Messages{emotion}.nlg'),
                self.domain)
        self.templates["neutral"] = TemplateFile(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'../../resources/nlg_templates/{self.domain.get_domain_name()}Messages.nlg'),
            self.domain)

        self._add_additional_methods_for_template_file()

    def _add_additional_methods_for_template_file(self):
        """add the function prefixed by "_template_" to the template file interpreter"""
        for (method_name, method) in inspect.getmembers(type(self), inspect.isfunction):
            if method_name.startswith('_template_'):
                for emotion in self.templates:
                    self.templates[emotion].add_python_function(method_name[10:], method, [self])
