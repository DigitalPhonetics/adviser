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

from typing import List, Iterable
from utils.domain.lookupdomain import LookupDomain
from examples.webapi.mensa.parser import MensaParser, DishType

SLOT_VALUES = {
    'day': ['today', 'tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
    'type': [DishType.Starter.value, DishType.Buffet.value, DishType.MainDish.value,
             DishType.SideDish.value, DishType.Dessert.value],
    'vegan': ['true', 'false'],
    'vegetarian': ['true', 'false'],
    'fish': ['true', 'false'],
    'pork': ['true', 'false'],
}


class MensaDomain(LookupDomain):
    """Domain for the Mensa API

    Attributes:
        parser (MensaParser): HTML file parser for dynamically building a pseudo database
        last_results (List[dict]): Current results which the user might request info about
    """

    def __init__(self):
        LookupDomain.__init__(self, 'MensaAPI', 'Mensa Food')
        self.parser = MensaParser()
        self.last_results = []

    def find_entities(self, constraints: dict, requested_slots: Iterable = iter(())):
        """ Returns all entities from the data backend that meet the constraints.

        Args:
            constraints (dict): Slot-value mapping of constraints.
                                If empty, all entities in the database will be returned.
            requested_slots (Iterable): list of slots that should be returned in addition to the
                                        system requestable slots and the primary key
        """
        if 'day' in constraints:
            meals = self.parser.get_meals(constraints['day'])
            results = [meal.as_dict() for meal in meals]
            for slot in constraints:
                if slot == 'day':
                    continue
                results = [candidate for candidate in results if candidate[slot] == constraints[slot]]
            for i, result in enumerate(results):
                result['artificial_id'] = i+1
            if list(requested_slots):
                cleaned_results = [{slot: result_dict[slot] for slot in requested_slots} for result_dict in results]
            else:
                cleaned_results = results
            self.last_results = results
            return cleaned_results
        else:
            return []

    def find_info_about_entity(self, entity_id: str, requested_slots: Iterable):
        """ Returns the values (stored in the data backend) of the specified slots for the
            specified entity.

        Args:
            entity_id (str): primary key value of the entity
            requested_slots (dict): slot-value mapping of constraints
        """
        result = {slot: self.last_results[int(entity_id)-1][slot] for slot in requested_slots}
        result['artificial_id'] = entity_id
        return [result]

    def get_requestable_slots(self) -> List[str]:
        """ Returns a list of all slots requestable by the user. """
        return ['name', 'type', 'price', 'allergens', 'vegan', 'vegetarian', 'fish', 'pork']

    def get_system_requestable_slots(self) -> List[str]:
        """ Returns a list of all slots requestable by the system. """
        return ['day', 'type', 'vegan', 'vegetarian', 'fish', 'pork']

    def get_informable_slots(self) -> List[str]:
        """ Returns a list of all informable slots. """
        return ['day', 'type', 'vegan', 'vegetarian', 'fish', 'pork']

    def get_mandatory_slots(self) -> List[str]:
        """ Returns a list of all mandatory slots. """
        return ['day']
        
    def get_default_inform_slots(self) -> List[str]:
        """ Returns a list of all default Inform slots. """
        return ['name']

    def get_possible_values(self, slot: str) -> List[str]:
        """ Returns all possible values for an informable slot

        Args:
            slot (str): name of the slot

        Returns:
            a list of strings, each string representing one possible value for
            the specified slot.
        """
        assert slot in SLOT_VALUES
        return SLOT_VALUES[slot]

    def get_primary_key(self) -> str:
        """ Returns the slot name that will be used as the 'name' of an entry """
        return 'artificial_id'

    def get_keyword(self):
        return 'mensa'
