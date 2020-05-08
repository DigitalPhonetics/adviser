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

import sys, json, os
import requests
import urllib.parse

from typing import List, Iterable
from utils.domain.lookupdomain import LookupDomain

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class WorldKnowledgeDomain(LookupDomain):
    # Attribute docstrings currently not considered by mkdocstrings -> write plugin?

    """
    Question answering for the world knowledge domain.
    
    Attributes:
        - artificial_id_counter (int): pseudo identifier for each entry
        - name_lex (Dict[str,str]): lexicon for matching topic's names to their KG entity
    """

    def __init__(self):
        """Calls super class' constructor and loads name lexicon"""

        LookupDomain.__init__(self, 'CSQA', 'World Knowledge')

        self.artificial_id_counter = 1 #int: lexicon for matching topic's names to their KG entity

        self.name_lex = self._init_name_lexicon()
        """Dict[str,str]: lexicon for matching topic's names to their KG entity."""

    def _init_name_lexicon(self):
        with open(os.path.join(get_root_dir(), 'resources', 'ontologies', 'qa', 'name_dict.json')) as f:
            return json.load(f)

    def _find_topic_entities(self, term):
        entities = self.name_lex.get(term, [])
        return [(ent['id'], ent['label']) for ent in entities]

    def _perform_out_query(self, relation, topic):
        url = 'https://query.wikidata.org/sparql'
        query = """ SELECT ?item ?itemLabel
                    WHERE
                    {
                    wd:%s wdt:%s ?item.
                    ?item rdfs:label ?itemLabel.
                    FILTER(LANG(?itemLabel) = 'en').
                    }""" % (topic, relation)
        body = 'query=(%s)' % urllib.parse.quote(query)
        body = {'query': query}
        # print(body, data_points)
        r = requests.post(url, params = {'format': 'json', 'content-type': 'application/sparql-query', 'user-agent': 'Python 3.6.8'}, data=body)
        # print('Status', r.status_code)
        data = r.json()
        return [res['itemLabel']['value'] for res in data['results'] ['bindings']]

    def _perform_in_query(self, relation, topic):
        url = 'https://query.wikidata.org/sparql'
        query = """ SELECT ?item ?itemLabel
                    WHERE
                    {
                    ?item wdt:%s wd:%s.
                    ?item rdfs:label ?itemLabel.
                    FILTER(LANG(?itemLabel) = 'en').
                    }""" % (relation, topic)
        body = 'query=(%s)' % urllib.parse.quote(query)
        body = {'query': query}
        r = requests.post(url, params = {
                'format': 'json',
                'content-type': 'application/sparql-query',
                'user-agent': 'Python 3.6.8'
            }, data=body)
        data = r.json()
        return [res['itemLabel']['value'] for res in data['results'] ['bindings']]

    def find_entities(self, constraints: dict):
        """ Returns all entities from the data backend that meet the constraints.

        Args:
            constraints (dict): slot-value mapping of constraints
        """
        assert 'relation' in constraints
        assert 'topic' in constraints
        assert 'direction' in constraints

        topics = self._find_topic_entities(constraints['topic'])
        if not topics:
            return []

        answers = []
        for topic_id, topic_label in topics:
            answer_ids = []
            if constraints['direction'] == 'out':
                answer_ids = self._perform_out_query(constraints['relation'], topic_id)
                for answer_id in answer_ids:
                    answers.append({
                        'subject': topic_label,
                        'predicate': constraints['relation'],
                        'object': answer_id
                    })
                    self.artificial_id_counter += 1
            else:
                answer_ids = self._perform_in_query(constraints['relation'], topic_id)
                for answer_id in answer_ids:
                    answers.append({
                        'subject': answer_id,
                        'predicate': constraints['relation'],
                        'object': topic_label
                    })
                    self.artificial_id_counter += 1

        return answers

    def find_info_about_entity(self, entity_id, requested_slots: Iterable):
        """ Returns the values (stored in the data backend) of the specified slots for the
            specified entity.

        Args:
            entity_id (str): primary key value of the entity
            requested_slots (dict): slot-value mapping of constraints
        """
        raise BaseException('should not be called')

    def get_domain_name(self):
        return "qa"

    def get_requestable_slots(self) -> List[str]:
        """ Returns a list of all slots requestable by the user. """
        return ['subject', 'predicate', 'object', 'object_type']

    def get_system_requestable_slots(self) -> List[str]:
        """ Returns a list of all slots requestable by the system. """
        return ['relation', 'topic', 'direction']

    def get_informable_slots(self) -> List[str]:
        """ Returns a list of all informable slots. """
        return ['relation', 'topic', 'direction']

    def get_mandatory_slots(self) -> List[str]:
        """ Returns a list of all mandatory slots. """
        return ['relation', 'topic', 'direction']

    def get_possible_values(self, slot: str) -> List[str]:
        """ Returns all possible values for an informable slot

        Args:
            slot (str): name of the slot

        Returns:
            a list of strings, each string representing one possible value for
            the specified slot.
        """
        # 'assert False, "this method should not be called"'
        raise BaseException('should not be called')

    def get_primary_key(self) -> str:
        """ Returns the slot name that will be used as the 'name' of an entry """
        return 'artificial_id'

    def get_keyword(self):
        return 'world knowledge'
