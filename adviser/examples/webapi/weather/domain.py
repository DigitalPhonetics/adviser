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
import json
from urllib.request import urlopen
from datetime import datetime

from utils.domain.lookupdomain import LookupDomain

API_KEY = 'EnterYourPersonalAPIKeyFromOpenWeatherMapOrg'


class WeatherDomain(LookupDomain):
    """Domain for the Weather API.

    Attributes:
        last_results (List[dict]): Current results which the user might request info about
    """

    def __init__(self):
        LookupDomain.__init__(self, 'WeatherAPI', 'Weather')

        self.last_results = []

    def find_entities(self, constraints: dict, requested_slots: Iterable = iter(())):
        """ Returns all entities from the data backend that meet the constraints.

        Args:
            constraints (dict): Slot-value mapping of constraints.
                                If empty, all entities in the database will be returned.
            requested_slots (Iterable): list of slots that should be returned in addition to the
                                        system requestable slots and the primary key
        """
        if 'location' in constraints and 'date' in constraints:

            forecast = self._query(constraints['location'], constraints['date'])
            if forecast is None:
                return []

            temperature = int('%.0f' % (float(forecast['main']['temp']) - 273.15))
            description = forecast['weather'][0]['description']

            result_dict = {
                'artificial_id': str(len(self.last_results)),
                'temperature': temperature,
                'description': description,
                'location': constraints['location'],
                'date': constraints['date'],
            }
            if any(True for _ in requested_slots):
                cleaned_result_dict = {slot: result_dict[slot] for slot in requested_slots}
            else:
                cleaned_result_dict = result_dict
            self.last_results.append(cleaned_result_dict)
            return [cleaned_result_dict]
        else:
            return []

    def find_info_about_entity(self, entity_id, requested_slots: Iterable):
        """ Returns the values (stored in the data backend) of the specified slots for the
            specified entity.

        Args:
            entity_id (str): primary key value of the entity
            requested_slots (dict): slot-value mapping of constraints
        """
        return [self.last_results[int(entity_id)]]

    def get_requestable_slots(self) -> List[str]:
        """ Returns a list of all slots requestable by the user. """
        return ['temperature', 'description']

    def get_system_requestable_slots(self) -> List[str]:
        """ Returns a list of all slots requestable by the system. """
        return ['location', 'date']

    def get_informable_slots(self) -> List[str]:
        """ Returns a list of all informable slots. """
        return ['location', 'date']

    def get_mandatory_slots(self) -> List[str]:
        """ Returns a list of all mandatory slots. """
        return ['location', 'date']
        
    def get_default_inform_slots(self) -> List[str]:
        """ Returns a list of all default Inform slots. """
        return ['temperature', 'description']

    def get_possible_values(self, slot: str) -> List[str]:
        """ Returns all possible values for an informable slot

        Args:
            slot (str): name of the slot

        Returns:
            a list of strings, each string representing one possible value for
            the specified slot.
         """
        raise BaseException('all slots in this domain do not have a fixed set of '
                            'values, so this method should never be called')

    def get_primary_key(self) -> str:
        """ Returns the slot name that will be used as the 'name' of an entry """
        return 'artificial_id'

    def _query(self, location, date):
        """if location is None:
            location = 'Stuttgart'
        if date is None:
            date = datetime.now()"""

        url = f'http://api.openweathermap.org/data/2.5/forecast?q={location}&APPID={API_KEY}'
        try:
            f = urlopen(url)
            forecasts = json.loads(f.read())['list']
            return self._find_closest(forecasts, date)
        except BaseException as e:
            raise(e)
            return None

    def _find_closest(self, forecasts, preferred_time):
        """ From a list of forecasts, find the one which is closest to the specified time"""
        closest_forecast = None
        closest_difference = None
        for forecast in forecasts:
            forecast_time = datetime.fromtimestamp(int(forecast['dt']))
            time_difference = abs(preferred_time - forecast_time)
            if closest_forecast is None or time_difference < closest_difference:
                closest_forecast = forecast
                closest_difference = time_difference
            else:
                return closest_forecast
        return None

    def get_keyword(self):
        return 'weather'
