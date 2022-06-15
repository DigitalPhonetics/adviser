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

from utils import DiasysLogger
from utils import SysAct, SysActionType
from services.service import Service, PublishSubscribe

import locale
locale.setlocale(locale.LC_TIME, 'en_GB.UTF-8')


class MovieNLG(Service):
    """Simple NLG for the weather domain"""

    def __init__(self, domain, logger=DiasysLogger()):
        # only calls super class' constructor
        super(MovieNLG, self).__init__(domain, debug_logger=logger)

    @PublishSubscribe(sub_topics=["sys_act"], pub_topics=["sys_utterance"])
    def generate_system_utterance(self, sys_act: SysAct = None) -> dict(sys_utterance=str):
        """Main function for generating and publishing the system utterance

        Args:
            sys_act: the system act for which to create a natural language realisation

        Returns:
            dict with "sys_utterance" as key and the system utterance as value
        """

        if sys_act is None or sys_act.type == SysActionType.Welcome:
            return {'sys_utterance': 'Hi! What do you want to know about movies?'}

        if sys_act.type == SysActionType.Bad:
            return {'sys_utterance': 'Sorry, I could not understand you.'}
        elif sys_act.type == SysActionType.Bye:
            return {'sys_utterance': 'Thank you, good bye'}
        elif sys_act.type == SysActionType.Request:
            slot = list(sys_act.slot_values.keys())[0]
            if slot == 'primary_release_year':
                return {'sys_utterance': 'For which year are you looking for a movie?'}
            elif slot == 'with_genres':
                return {'sys_utterance': 'Which genre are you interested in?'}
            else:
                assert False, 'Only a year and a genre can be requested'
        else:
            with_genres = sys_act.slot_values['with_genres'][0]
            primary_release_year = sys_act.slot_values['primary_release_year'][0]
            title = sys_act.slot_values['title'][0]
            overview = sys_act.slot_values['overview'][0]
            return {'sys_utterance': f'A {with_genres} movie released in {primary_release_year} is called {title}: {overview}.'}
