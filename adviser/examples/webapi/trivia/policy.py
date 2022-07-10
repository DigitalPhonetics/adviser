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

from typing import List, Dict

from utils.domain.lookupdomain import LookupDomain
from services.service import PublishSubscribe, Service
from utils import SysAct, SysActionType
from utils.logger import DiasysLogger
from utils.useract import UserAct, UserActionType
from collections import defaultdict


class TriviaPolicy(Service):
    """Policy module for question answering.

    Provides a simple rule-based policy for question answering.
    The QA module assumes that the user acts contain information about relation, topic entities
    and relation direction.
    Adequate answers are looked up in the knowledge and published.

    The difference to the default HandcraftedPolicy is that no BST is needed and that multiple
    system acts can be published.
    """
    def __init__(self, domain: LookupDomain, logger: DiasysLogger = DiasysLogger()):
        # only call super class' constructor
        Service.__init__(self, domain=domain, debug_logger=logger)

    @PublishSubscribe(sub_topics=["user_acts"], pub_topics=["sys_acts", "sys_state"])
    def generate_sys_acts(self, user_acts: List[UserAct] = None) -> dict(sys_acts=List[SysAct]):
        if user_acts is None:
            return { 'sys_acts': [SysAct(SysActionType.Welcome)]}
        elif any([user_act.type == UserActionType.Bye for user_act in user_acts]):
            return { 'sys_acts': [SysAct(SysActionType.Bye)] }
        elif not user_acts:
            return { 'sys_acts': [SysAct(SysActionType.Bad)] }
        
        user_acts = [user_act for user_act in user_acts if user_act.type != UserActionType.SelectDomain]
        if len(user_acts) == 0:
           return { 'sys_acts': [SysAct(SysActionType.Welcome)]}
        
        entities_constraints = {}
        for user_act in user_acts:
            if user_act.type == UserActionType.Inform:
                if user_act.slot == 'level':
                    entities_constraints.update({
                        'level': user_act.value
                    })
                if user_act.slot == 'quiztype':
                    entities_constraints.update({
                        'quiztype': user_act.value
                    })
                if user_act.slot == 'category':
                    entities_constraints.update({
                        'category': user_act.value
                    })

        question = self.domain.find_entities(entities_constraints)
        sys_act = SysAct(SysActionType.TellQuestion, slot_values={'question': question['question']})

        self.debug_logger.dialog_turn("System Action: " + '; ' + str(sys_act))
        return {'sys_acts': [sys_act]}