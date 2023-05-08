###############################################################################
#
# Copyright 2023, University of Stuttgart: Institute for Natural Language Processing (IMS)
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
# External dependencies:
# This file uses code and model checkpoints from https://github.com/ConvLab/ConvLab-3,
# which is licensed under Apache License 2.0.
#
###############################################################################

from typing import List
from examples.ml_nlu.t5nlu import T5NLU
from utils.useract import UserActionType
from utils.useract import UserAct
from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.service import Service, PublishSubscribe



class MLNLU(Service):
    """
    Example of how to integrate a Machine-Learning based NLU as a service.
    """

    def __init__(self, domain: JSONLookupDomain):
        # domain here is only needed to setup the correct topics 
        # (appends the domain name to each subscribed / published topic within this class)
        Service.__init__(self, domain=domain)

        # load t5-small based pre-trained NLU model from https://github.com/ConvLab/ConvLab-3/tree/master/convlab/base_models/t5
        self.model = T5NLU(speaker='user', context_window_size=0, model_name_or_path='ConvLab/t5-small-nlu-multiwoz21')


    def dialog_start(self) -> dict:
        """
        Sets the previous system act as None.
        This function is called when the dialog starts

        Returns:
            Empty dictionary

        """
        self.sys_act_info = {
            'last_act': None, 'lastInformedPrimKeyVal': None, 'lastRequestSlot': None}
        self.user_acts = []
        self.slots_informed = set()
        self.slots_requested = set()
        self.req_everything = False

    @PublishSubscribe(sub_topics=["user_utterance"], pub_topics=["user_acts"])
    def extract_user_acts(self, user_utterance: str = None) -> dict(user_acts=List[UserAct]):

        """
        Responsible for detecting user acts with their respective slot-values from the user
        utterance through regular expressions.

        Args:
            user_utterance (BeliefState) - a BeliefState obejct representing current system
                                           knowledge

        Returns:
            dict of str: UserAct - a dictionary with the key "user_acts" and the value
                                            containing a list of user actions
        """
        
        # acts from t5 model are in form of a list of recognized acts
        # - each act is a list of strings, where
        #   1. intent
        #   2. domain
        #   3. slot
        #   4. value
        # Example: "I want a greek Restaurant" -> [['inform', 'restaurant', 'food', 'greek']]
        raw_acts = self.model.predict(user_utterance)

        # convert to ADVISER user act format (this is not exhaustive, just as a quick example)
        user_acts = []
        for act in raw_acts:
            # decompose list entry
            intent, domain, slot, value = act
            if intent in UserActionType._value2member_map_:
                # if the intent is a an intent we know, create a user action from it
                user_acts.append(UserAct(text=user_utterance,
                                        act_type=UserActionType(intent),
                                        slot=slot,
                                        value=value))
                print("Recognized user act:", user_acts)
            else:
                print("Unrecognized user act: ", act)
        
        return {"user_acts": user_acts}

