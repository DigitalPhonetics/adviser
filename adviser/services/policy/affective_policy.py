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

from typing import Dict

from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.service import PublishSubscribe
from services.service import Service
from utils.logger import DiasysLogger
from utils.userstate import UserState


class EmotionPolicy(Service):
    """ Module for deciding what type of emotional response/ engagement level of response, the system
        should give

    """
    def __init__(self, domain: JSONLookupDomain = None, logger: DiasysLogger = DiasysLogger()):
        """
        Initializes the policy

        Arguments:
            domain (JSONLookupDomain): the domain that the affective policy should operate in

        """
        self.first_turn = True
        Service.__init__(self, domain=domain)
        self.logger = logger

    def dialog_start(self):
        pass

    @PublishSubscribe(sub_topics=["userstate"], pub_topics=["sys_emotion", "sys_engagement"])
    def choose_sys_emotion(self, userstate: UserState = None)\
            -> Dict[str, str]:

        """
            This method maps observed user emotion and user engagement to the system's choices
            for output emotion/engagement

            Args:
                userstate (UserState): a UserState obejct representing current system
                                       knowledge of the user's emotional state and engagement

            Returns:
                (dict): a dictionary with the keys "sys_emotion" and "sys_engagement" and the
                        corresponding values

        """
        return {"sys_emotion": userstate["emotion"]["category"].value,
                "sys_engagement": userstate["engagement"].value}
