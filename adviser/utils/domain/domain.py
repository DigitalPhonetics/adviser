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


class Domain(object):
    """ Abstract class for linking a domain with a data access method.

        Derive from this class if you need to implement a domain with a not yet
        supported data backend, otherwise choose a fitting existing child class. """

    def __init__(self, name: str):
        self.name = name

    def get_domain_name(self) -> str:
        """ Return the domain name of the current ontology.

        Returns:
            object:
        """
        return self.name

    def find_entities(self, constraints : dict):
        """ Returns all entities from the data backend that meet the constraints.

        Args:
            constraints (dict): slot-value mapping of constraints

        IMPORTANT: This function must be overridden!
        """
        raise NotImplementedError
