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

from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.service import PublishSubscribe
from services.service import Service
from utils.logger import DiasysLogger
from utils.useract import UserAct
from utils.userstate import EmotionType, EngagementType
import random
from typing import List, Dict


class EmotionSimulator(Service):
    """
        Class which generates user emotion/engagements. Currently outputs either a user defined
        or random emotion/engagement level and was designed to test the affective services
        work correctly. However, in the future it could be extended to be more realistic.
    """
    def __init__(self, domain: JSONLookupDomain = None, logger: DiasysLogger = None,
                 random: bool = True, static_emotion: EmotionType = EmotionType.Neutral,
                 static_engagement: EngagementType = EngagementType.High):
        Service.__init__(self, domain=domain)
        self.domain = domain
        self.logger = logger
        self.random = random
        self.engagement = static_engagement
        self.emotion = static_emotion

    @PublishSubscribe(sub_topics=["user_acts"], pub_topics=["emotion", "engagement"])
    def send_emotion(self, user_acts: List[UserAct] = None) -> Dict[str, str]:
        """
            Publishes an emotion and engagement value for a turn

            Args:
                user_acts (List[UserAct]): the useracts, needed to synchronize when emotion should
                                           be generated

            Returns:
                (dict): A dictionary representing the simulated user emotion and engagement. The keys are
                        "emotion" and "engagement" where "emotion" is a dictionary which currently only contains
                        emotion category information but could be expanded to include other emotion measures, and
                        "engagement" corresponds to and EngagementType object.
        """
        if not self.random:
            return {"emotion": {"category": self.emotion}, "engagement": self.engagement}

        else:
            emotion = random.choice([e for e in EmotionType])
            engagement = random.choice([e for e in EngagementType])
            return {"emotion": {"category": emotion}, "engagement": engagement}
