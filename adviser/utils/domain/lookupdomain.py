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
from utils.domain.domain import Domain

class LookupDomain(Domain):
    """ Abstract class for linking a domain with a data access method.

        Derive from this class if you need to implement a domain with a not yet
        supported data backend, otherwise choose a fitting existing child class. """

    def __init__(self, identifier : str, display_name : str):
        Domain.__init__(self, identifier)
        self.display_name = display_name

    def find_entities(self, constraints : dict, requested_slots: Iterable = iter(())):
        """ Returns all entities from the data backend that meet the constraints.

        Args:
            constraints (dict): slot-value mapping of constraints

        IMPORTANT: This function must be overridden!
        """
        raise NotImplementedError

    def find_info_about_entity(self, entity_id, requested_slots: Iterable):
        """ Returns the values (stored in the data backend) of the specified slots for the
            specified entity.

        Args:
            entity_id (str): primary key value of the entity
            requested_slots (dict): slot-value mapping of constraints
        """
        raise NotImplementedError

    def get_display_name(self):
        return self.display_name

    def get_requestable_slots(self) -> List[str]:
        """ Returns a list of all slots requestable by the user. """
        raise NotImplementedError

    def get_system_requestable_slots(self) -> List[str]:
        """ Returns a list of all slots requestable by the system. """
        raise NotImplementedError

    def get_informable_slots(self) -> List[str]:
        """ Returns a list of all informable slots. """
        raise NotImplementedError

    def get_mandatory_slots(self) -> List[str]:
        """Returns a list of all mandatory slots.
        
        Slots are called mandatory if their value is required by the system before it can even
        generate a candidate list.
        """
        raise NotImplementedError

    def get_default_inform_slots(self) -> List[str]:
        """Returns a list of all default Inform slots.
        
        Default Inform slots are always added to (system) Inform actions, even if the user has not
        implicitly asked for it. Note that these slots are different from the primary key slot.
        """
        raise NotImplementedError

    def get_possible_values(self, slot: str) -> List[str]:
        """ Returns all possible values for an informable slot

        Args:
            slot (str): name of the slot

        Returns:
            a list of strings, each string representing one possible value for
            the specified slot.
         """
        raise NotImplementedError

    def get_primary_key(self) -> str:
        """ Returns the slot name that will be used as the 'name' of an entry """
        raise NotImplementedError

    def get_keyword(self):
        raise NotImplementedError
