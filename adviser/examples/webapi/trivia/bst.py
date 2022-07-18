from typing import List, Set

from services.service import PublishSubscribe
from services.service import Service
from utils.beliefstate import BeliefState
from utils.useract import UserActionType, UserAct
from utils import SysAct


class TriviaBST(Service):

    def __init__(self, domain=None, logger=None):
        Service.__init__(self, domain=domain)
        self.logger = logger
        self.bs = BeliefState(domain)

    @PublishSubscribe(sub_topics=["user_acts"], pub_topics=["beliefstate"])
    def update_bst(
        self,
        user_acts: List[UserAct] = None,
    ) -> dict(beliefstate=BeliefState):
        self.bs.start_new_turn()
        if user_acts:
            self._reset_informs(user_acts)
            self._reset_requests()
            self.bs["user_acts"] = self._get_all_usr_action_types(user_acts)
            self._handle_user_acts(user_acts)

        return {'beliefstate': self.bs}

    def dialog_start(self):
        self.bs = BeliefState(self.domain)

    def _reset_informs(self, acts: List[UserAct]):
        slots = {act.slot for act in acts if act.type == UserActionType.Inform}
        for slot in [s for s in self.bs['informs']]:
            if slot in slots:
                del self.bs['informs'][slot]

    def _reset_requests(self):
        self.bs['requests'] = {}

    def _get_all_usr_action_types(self, user_acts: List[UserAct]) -> Set[UserActionType]:
        action_type_set = set()
        for act in user_acts:
            action_type_set.add(act.type)
        return action_type_set

    def _handle_user_acts(self, user_acts: List[UserAct]):        
        # reset any offers if the user informs any new information
        if self.domain.get_primary_key() in self.bs['informs'] \
                and UserActionType.Inform in self.bs["user_acts"]:
            del self.bs['informs'][self.domain.get_primary_key()]

        # We choose to interpret switching as wanting to start a new dialog and do not support
        # resuming an old dialog
        elif UserActionType.SelectDomain in self.bs["user_acts"]:
            self.bs["informs"] = {}
            self.bs["requests"] = {}

        # Handle user acts
        for act in user_acts:
            if act.type == UserActionType.Request:
                self.bs['requests'] = act.slot
            elif act.type == UserActionType.Inform:
                # add informs and their scores to the beliefstate
                self.bs['informs'][act.slot] = act.value
            elif act.type == UserActionType.NegativeInform:
                # reset mentioned value to zero probability
                if act.slot in self.bs['informs']:
                    if act.value in self.bs['informs'][act.slot]:
                        del self.bs['informs'][act.slot][act.value]
            elif act.type == UserActionType.RequestAlternatives:
                # This way it is clear that the user is no longer asking about that one item
                if self.domain.get_primary_key() in self.bs['informs']:
                    del self.bs['informs'][self.domain.get_primary_key()]
