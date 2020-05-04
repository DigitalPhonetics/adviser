###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
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

import inspect
import os

from services.nlg.nlg import HandcraftedNLG
from services.service import PublishSubscribe
from utils.common import Language
from utils.domain.domain import Domain
from utils.logger import DiasysLogger
from utils.sysact import SysAct
from typing import Dict, List


class MultiNLG(HandcraftedNLG):
    """Extension of the handcrafted NLG by allowing multiple system acts.

    This change is necessary for QA, since the policy publishes multiple system acts.
    """
    def __init__(self, domain: Domain, template_file: str = None, sub_topic_domains: Dict[str, str] = {},
                 logger: DiasysLogger = DiasysLogger(), template_file_german: str = None,
                 language: Language = None):
        # only calls the super class' constructor
        HandcraftedNLG.__init__(self, domain, template_file, sub_topic_domains, logger, template_file_german, language)

    @PublishSubscribe(sub_topics=["sys_acts"], pub_topics=["sys_utterance"])
    def publish_system_utterance(self, sys_acts: List[SysAct] = None) -> dict(sys_utterance=str):
        """Generates the system utterance and publishes it.

        Instead of the HandcraftedNLG class, this class takes multiple system acts and
        applies the templates for each system act separately. Each message is then printed in a
        new line.

        Args:
            sys_acts: The list of system acts as published by the (QA) policy

        Returns:
            a dict containing the system utterance
        """
        message = '\n'.join([self.generate_system_utterance(sys_act) for sys_act in sys_acts])
        return {'sys_utterance': message}
