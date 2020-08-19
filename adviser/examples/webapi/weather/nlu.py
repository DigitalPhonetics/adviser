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

from utils import UserAct, UserActionType, DiasysLogger, SysAct, BeliefState
from services.service import Service, PublishSubscribe

# simple list of regexes

WEATHER_DATE_TODAY_REGEXES = [
    re.compile(r'\btoday\b|\bnow\b|\bcurrently\b')
]

WEATHER_DATE_TOMORROW_REGEXES = [
    re.compile(r'\btomorrow\b')
]

WEATHER_LOCATION_REGEXES = [
    re.compile(r'\bin ([^ ]*)\b')
]


class WeatherNLU(Service):
    """Very simple NLU for the weather domain."""

    def __init__(self, domain, logger=DiasysLogger()):
        # only calls super class' constructor
        super(WeatherNLU, self).__init__(domain, debug_logger=logger)

    @PublishSubscribe(sub_topics=["user_utterance"], pub_topics=["user_acts"])
    def extract_user_acts(self, user_utterance: str = None) -> dict(user_acts=List[UserAct]):
        """Main function for detecting and publishing user acts.

        Args:
            user_utterance: the user input string

        Returns:
            dict with key 'user_acts' and list of user acts as value
        """
        user_acts = []
        if not user_utterance:
            return {'user_acts': None}
        user_utterance = ' '.join(user_utterance.lower().split())

        for bye in ('bye', 'goodbye', 'byebye', 'seeyou'):
            if user_utterance.replace(' ', '').endswith(bye):
                return {'user_acts': [UserAct(user_utterance, UserActionType.Bye)]}

        # check weather today
        for regex in WEATHER_DATE_TODAY_REGEXES:
            match = regex.search(user_utterance)
            if match:
                user_acts.append(UserAct(user_utterance, UserActionType.Inform, 'date', datetime.now()))
                break
        if len(user_acts) == 0:
            for regex in WEATHER_DATE_TOMORROW_REGEXES:
                match = regex.search(user_utterance)
                if match:
                    tomorrow = datetime.now() + timedelta(days=1)
                    date = datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour=15)
                    user_acts.append(UserAct(user_utterance, UserActionType.Inform, 'date', date))
                    break
        for regex in WEATHER_LOCATION_REGEXES:
            match = regex.search(user_utterance)
            if match:
                user_acts.append(UserAct(user_utterance, UserActionType.Inform, 'location', match.group(1)))

        self.debug_logger.dialog_turn("User Actions: %s" % str(user_acts))
        return {'user_acts': user_acts}
