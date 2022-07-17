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

import json
import ssl
import random
from typing import List, Iterable
from utils.domain.lookupdomain import LookupDomain
from urllib.request import urlopen

categories = {
    'general': [9],
    'entertainment': [10,11,12,13,14,15,16,26,29,31,32],
    'science': [17,18,19,27,30],
    'society': [20,21,22,23,24,25,28]
}

class TriviaDomain(LookupDomain):
    """Domain for the Trivia API"""

    def __init__(self):
        LookupDomain.__init__(self, 'Trivia', 'Trivia')
        self.count = 0
        self.score = 0
        self.correct_answer = None
        self.question = None

    def find_entities(self, constraints: dict, requested_slots: Iterable = iter(())):
        level = constraints['difficulty_level'] if 'difficulty_level' in constraints else 'easy'
        quiztype = constraints['quiztype'] if 'quiztype' in constraints else 'boolean'
        category = constraints['category'] if 'category' in constraints else 'general'
        length='5'

        trivia_instance = self._query(
            level = level,
            quiztype =  quiztype,
            category =  category
            )

        if trivia_instance is None:
            return []

        self.correct_answer = True if trivia_instance['results'][0]['correct_answer'] == "True" else False
        self.question = trivia_instance['results'][0]['question']
        
        self.count += 1
        
        return [{
            'artificial_id': 1, 'level': level, 'quiztype': quiztype,
            'category': category, 'length': length
        }]

    def find_info_about_entity(self, entity_id: str, requested_slots: Iterable):
        result = {slot: self.last_results[int(entity_id)-1][slot] for slot in requested_slots}
        result['artificial_id'] = entity_id
        return [result]

    def get_requestable_slots(self) -> List[str]:
        return ['true', 'false']

    def get_system_requestable_slots(self) -> List[str]:
        return ['answer', 'score', 'counter']

    def get_informable_slots(self) -> List[str]:
        return ['level', 'category', 'quiztype', 'length']

    def get_mandatory_slots(self) -> List[str]:
        return ['given_answer']
        
    def get_default_inform_slots(self) -> List[str]:
        return []

    def get_possible_values(self, slot: str) -> List[str]:
        assert slot in SLOT_VALUES
        return SLOT_VALUES[slot]

    def get_primary_key(self) -> str:
        return 'artificial_id'

    def _query(self, level, quiztype, category):
        url = f'https://opentdb.com/api.php?amount=1&difficulty={level}' \
            f'&type={quiztype}&category={random.choice(categories[category])}'
        try:
            context = ssl._create_unverified_context()
            f = urlopen(url, context=context)
            instance = json.loads(f.read())
            return instance
        except BaseException as e:
            raise(e)
            return None
    
    def get_keyword(self):
        return 'trivia'
