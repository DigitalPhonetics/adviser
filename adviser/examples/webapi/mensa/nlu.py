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

from utils.logger import DiasysLogger
from utils.domain.lookupdomain import LookupDomain
from utils import UserAct, UserActionType
from utils.sysact import SysAct
from utils.beliefstate import BeliefState
from utils.common import Language
from services.service import PublishSubscribe

from services.nlu import HandcraftedNLU


class MensaNLU(HandcraftedNLU):
    """Adapted handcrafted NLU for the mensa domain.

    The default handcrafted NLU is adapted to automatically add the user act request(name).
    This is necessary because the name is not the primary key, i.e. it is not printed by default
    once an element is found. To force the Policy to automatically inform about the name, too,
    a request for the name is added in each turn.
    """

    def __init__(self, domain: LookupDomain, logger: DiasysLogger = DiasysLogger()):
        # only calls super class' constructor
        HandcraftedNLU.__init__(self, domain, logger)

    @PublishSubscribe(sub_topics=["user_utterance"], pub_topics=["user_acts"])
    def extract_user_acts(self, user_utterance: str = None, sys_act: SysAct = None, beliefstate: BeliefState = None) \
            -> dict(user_acts=List[UserAct]):
        """Original code but adapted to automatically add a request(name) act"""
        result = {}

        # Setting request everything to False at every turn
        self.req_everything = False

        self.user_acts = []

        # slots_requested & slots_informed store slots requested and informed in this turn
        # they are used later for later disambiguation
        self.slots_requested, self.slots_informed = set(), set()
        if user_utterance is not None:
            user_utterance = user_utterance.strip()
            self._match_general_act(user_utterance)
            self._match_domain_specific_act(user_utterance)

        # Solving ambiguities from regexes, especially with requests and informs happening
        # simultaneously on the same slot and two slots taking the same value
        self._disambiguate_co_occurrence(beliefstate)
        self._solve_informable_values()

        # If nothing else has been matched, see if the user chose a domain; otherwise if it's
        # not the first turn, it's a bad act
        if len(self.user_acts) == 0:
            if self.domain.get_keyword() in user_utterance:
                self.user_acts.append(UserAct(text=user_utterance if user_utterance else "",
                                              act_type=UserActionType.SelectDomain))
            elif self.sys_act_info['last_act'] is not None:
                # start of dialogue or no regex matched
                self.user_acts.append(UserAct(text=user_utterance if user_utterance else "",
                                              act_type=UserActionType.Bad))

        self._assign_scores()
        result['user_acts'] = self.user_acts
        self.logger.dialog_turn("User Actions: %s" % str(self.user_acts))
        return result
