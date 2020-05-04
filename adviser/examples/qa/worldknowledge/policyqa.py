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


class QaPolicy(Service):
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

    @PublishSubscribe(sub_topics=["user_acts"], pub_topics=["sys_acts"])
    def generate_sys_acts(self, user_acts: List[UserAct] = None) -> dict(sys_acts=List[SysAct]):
        """Generates system acts by looking up answers to the given user question.

        Args:
            user_acts: The list of user acts containing information about the predicted relation,
                topic entities and relation direction

        Returns:
            dict with 'sys_acts' as key and list of system acts as value
        """
        if user_acts is None:
            return { 'sys_acts': [SysAct(SysActionType.Welcome)]}
        elif any([user_act.type == UserActionType.Bye for user_act in user_acts]):
            return { 'sys_acts': [SysAct(SysActionType.Bye)] }
        elif not user_acts:
            return { 'sys_acts': [SysAct(SysActionType.Bad)] }
        
        user_acts = [user_act for user_act in user_acts if user_act.type != UserActionType.SelectDomain]
        if len(user_acts) == 0:
           return { 'sys_acts': [SysAct(SysActionType.Welcome)]} 

        relation = [user_act.value for user_act in user_acts \
            if user_act.type == UserActionType.Inform and user_act.slot == 'relation'][0]
        topics = [user_act.value for user_act in user_acts \
            if user_act.type == UserActionType.Inform and user_act.slot == 'topic']
        direction = [user_act.value for user_act in user_acts \
            if user_act.type == UserActionType.Inform and user_act.slot == 'direction'][0]

        if not topics:
            return { 'sys_acts': [SysAct(SysActionType.Bad)] }

        # currently, short answers are used for world knowledge
        answers = self._get_short_answers(relation, topics, direction)

        sys_acts = [SysAct(SysActionType.InformByName, slot_values=answer) for answer in answers]

        self.debug_logger.dialog_turn("System Action: " + '; '.join(
            [str(sys_act) for sys_act in sys_acts]))
        return {'sys_acts': sys_acts}

    def _get_short_answers(self, relation: str, topics: List[str], direction: str) \
        -> dict(answer=str):
        """Looks up answers and only returns the answer string"""
        answers = []
        for topic in topics:
            triples = self.domain.find_entities({
                'relation': relation,
                'topic': topic,
                'direction': direction
            })
            for triple in triples:
                if direction == 'in':
                    answers.append({'answer': triple['subject']})
                else:
                    answers.append({'answer': triple['object']})
        if not answers:
            answers.append({'answer': 'Sorry, I don\'t know.'})
        return answers

    def _get_triples(self, relation, topics, direction):
        """Looks up answers and stores them as triples"""
        answers = []
        for topic in topics:
            answers.extend(self.domain.find_entities({
                'relation': relation,
                'topic': topic,
                'direction': direction
            }))
        if not answers:
            for topic in topics:
                answers.append({
                    'subject': 'unknown' if direction == 'in' else topic,
                    'predicate': relation,
                    'object': 'unknown' if direction == 'out' else topic
                })
        return answers
