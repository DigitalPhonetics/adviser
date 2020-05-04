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

from services.service import Service
from services.service import PublishSubscribe
from utils.userstate import EmotionType, EngagementType, UserState


class HandcraftedUST(Service):
    """
    A rule-based approach on user state tracking. Currently very minimalist
    """

    def __init__(self, domain=None, logger=None):
        Service.__init__(self, domain=domain)
        self.logger = logger
        self.us = UserState()

    @PublishSubscribe(sub_topics=["emotion", "engagement"], pub_topics=["userstate"])
    def update_emotion(self, emotion: EmotionType = None, engagement: EngagementType = None) \
            -> dict(userstate=UserState):
        """
            Function for updating the userstate (which tracks the system's knowledge about the
            user's emotions/engagement

            Args:
                emotion (EmotionType): what emotion has been identified for the user
                engagement (list): a list of UserAct objects mapped from the user's last utterance

            Returns:
                (dict): a dictionary with the key "userstate" and the value a UserState object

        """
        # save last turn to memory
        self.us.start_new_turn()
        self.us["engagement"] = engagement
        self.us["emotion"] = emotion
        return {'userstate': self.us}

    def dialog_start(self):
        """
            Resets the user state so it is ready for a new dialog
        """
        # initialize belief state
        self.us = UserState()
