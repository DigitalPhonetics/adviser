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

from typing import Any, Dict, List, Set
from utils.domain.domain import Domain
from utils.memory import UserState

from services.service import PublishSubscribe
from services.service import Service
from utils.beliefstate import BeliefState
from utils.useract import UserActionType, UserAct


class HandcraftedBST(Service):
    """
    A rule-based approach to belief state tracking.
    """

    def __init__(self, domain: Domain=None, logger=None, identifier="BST_Handcrafted",
                 transports: str = "ws://localhost:8080/ws", realm="adviser") -> None:
        Service.__init__(self, domain=domain, identifier=identifier, transports=transports, realm=realm)
        self.logger = logger
        self.bs = UserState(lambda: BeliefState(domain))

    @PublishSubscribe(sub_topics=["user_acts"], pub_topics=["beliefstate"], user_id=True)
    def update_bst(self, user_acts: List[Dict[str, Any]], user_id: int) -> dict(beliefstate=BeliefState):
        """
            Updates the current dialog belief state (which tracks the system's
            knowledge about what has been said in the dialog) based on the user actions generated
            from the user's utterances

            Args:
                user_acts (list): a list of UserAct objects mapped from the user's last utterance

            Returns:
                (dict): a dictionary with the key "beliefstate" and the value the updated
                        BeliefState object

        """
        # save last turn to memory
        user_acts = [UserAct.from_json(act) for act in user_acts]
        self.bs[user_id].start_new_turn()
        if user_acts:
            self._reset_informs(user_acts, user_id)
            self._reset_requests(user_id)
            self.bs[user_id]["user_acts"] = self._get_all_usr_action_types(user_acts)

            self._handle_user_acts(user_acts, user_id)

            num_entries, discriminable = self.bs[user_id].get_num_dbmatches()
            self.bs[user_id]["num_matches"] = num_entries
            self.bs[user_id]["discriminable"] = discriminable

        return {'beliefstate': self.bs[user_id]}


    async def on_dialog_start(self, user_id: int):
        """
            Restets the belief state so it is ready for a new dialog

            Returns:
                (dict): a dictionary with a single entry where the key is 'beliefstate'and
                        the value is a new BeliefState object
        """
        # initialize belief state
        self.bs[user_id] = BeliefState(self.domain)

    def _reset_informs(self, acts: List[UserAct], user_id: int):
        """
            If the user specifies a new value for a given slot, delete the old
            entry from the beliefstate
        """

        slots = {act.slot for act in acts if act.type == UserActionType.Inform}
        for slot in [s for s in self.bs[user_id]['informs']]:
            if slot in slots:
                del self.bs[user_id]['informs'][slot]

    def _reset_requests(self, user_id: int):
        """
            gets rid of requests from the previous turn
        """
        self.bs[user_id]['requests'] = {}

    def _get_all_usr_action_types(self, user_acts: List[UserAct]) -> Set[UserActionType]:
        """ 
        Returns a set of all different UserActionTypes in user_acts.

        Args:
            user_acts (List[UserAct]): list of UserAct objects

        Returns:
            set of UserActionType objects
        """
        action_type_set = set()
        for act in user_acts:
            action_type_set.add(act.type)
        return action_type_set

    def _handle_user_acts(self, user_acts: List[UserAct], user_id: int):

        """
            Updates the belief state based on the information contained in the user act(s)

            Args:
                user_acts (list[UserAct]): the list of user acts to use to update the belief state

        """
        
        # reset any offers if the user informs any new information
        if self.domain.get_primary_key() in self.bs[user_id]['informs'] \
                and UserActionType.Inform in self.bs[user_id]["user_acts"]:
            del self.bs[user_id]['informs'][self.domain.get_primary_key()]

        # We choose to interpret switching as wanting to start a new dialog and do not support
        # resuming an old dialog
        elif UserActionType.SelectDomain in self.bs[user_id]["user_acts"]:
            self.bs[user_id]["informs"] = {}
            self.bs[user_id]["requests"] = {}

        # Handle user acts
        for act in user_acts:
            if act.type == UserActionType.Request:
                self.bs[user_id]['requests'][act.slot] = act.score
            elif act.type == UserActionType.Inform:
                # add informs and their scores to the beliefstate
                if act.slot in self.bs[user_id]["informs"]:
                    self.bs[user_id]['informs'][act.slot][act.value] = act.score
                else:
                    self.bs[user_id]['informs'][act.slot] = {act.value: act.score}
            elif act.type == UserActionType.NegativeInform:
                # reset mentioned value to zero probability
                if act.slot in self.bs[user_id]['informs']:
                    if act.value in self.bs[user_id]['informs'][act.slot]:
                        del self.bs[user_id]['informs'][act.slot][act.value]
            elif act.type == UserActionType.RequestAlternatives:
                # This way it is clear that the user is no longer asking about that one item
                if self.domain.get_primary_key() in self.bs[user_id]['informs']:
                    del self.bs[user_id]['informs'][self.domain.get_primary_key()]
