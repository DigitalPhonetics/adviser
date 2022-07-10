############################################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify'
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
############################################################################################

import re
from datetime import datetime, timedelta
from typing import List

from utils.logger import DiasysLogger
from utils.domain.lookupdomain import LookupDomain
from utils import UserAct, UserActionType
from utils.sysact import SysAct
from utils.beliefstate import BeliefState
from services.service import Service, PublishSubscribe
from services.nlu import HandcraftedNLU


class TriviaNLU(HandcraftedNLU):
    """Very simple NLU for the trivia domain."""

    def __init__(self, domain: LookupDomain, logger: DiasysLogger = DiasysLogger()):
        # only calls super class' constructor
        HandcraftedNLU.__init__(self, domain, logger)

    @PublishSubscribe(sub_topics=["user_utterance"], pub_topics=["user_acts"])
    def extract_user_acts(
        self,
        user_utterance: str = None,
        sys_act: SysAct = None,
        beliefstate: BeliefState = None
    ) -> dict(user_acts=List[UserAct]):
        """Main function for detecting and publishing user acts.

        Args:
            user_utterance: the user input string

        Returns:
            dict with key 'user_acts' and list of user acts as value
        """
        result = {}
        self.user_acts = []
        self.slots_requested, self.slots_informed = set(), set()
                
        if not user_utterance:
            return {'user_acts': None}
        else:
            user_utterance = user_utterance.strip()
            self._match_general_act(user_utterance)
            self._match_domain_specific_act(user_utterance)

        self.logger.dialog_turn("User Actions: %s" % str(self.user_acts))
        return {'user_acts': self.user_acts}
