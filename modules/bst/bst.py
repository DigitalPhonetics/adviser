###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
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

from modules.module import Module
from utils.beliefstate import BeliefState
from utils.useract import UserActionType, UserAct
from utils import SysAct, SysActionType
from typing import List
from utils.logger import DiasysLogger


class HandcraftedBST(Module):
    """
    A rule-based approach on belief state tracking.
    The state is basically a dictionary of keys.
    """

    def __init__(self, domain=None, subgraph=None,
                 logger: DiasysLogger = DiasysLogger()):
        super(HandcraftedBST, self).__init__(domain, subgraph, logger=logger)
        # Informables and Requestables are extracted and provided with
        # probabilities e.g. state['beliefs']['food']['italian'] = 0.0
        self.inform_scores = {}
        self.request_slots = {}
        self.primary_key = domain.get_primary_key()


    def start_dialog(self, **kwargs):
        """
            Restets the belief state so it is ready for a new dialog

            Returns:
                (dict): a dictionary with a single entry where the key is 'beliefstate'and
                        the value is a new BeliefState object
        """
        # initialize belief state
        self.inform_scores = {}
        self.request_slots = {}
        return {'beliefstate': BeliefState(self.domain)}


    def _zero_all_scores(self, slot_values: dict):
        """
            Sets all scores of the slot-value dict to 0.0

            Args:
                slot_values (dict): a dictionary where the keys are slots and the values
                                    are floats representing the systems confidence that the
                                    user has specified that slot
        """
        for slot in slot_values:
            slot_values[slot] = 0.0


    def _get_all_usr_action_types(self, user_acts: List[UserAct]):
        """ Returns a set of all different user action types in user_acts.

        Args:
            user_acts: list of UsrAct objects

        Returns:
            set of UserActionType objects
        """
        action_type_list = set()
        for act in user_acts:
            action_type_list.add(act.type)
        return action_type_list


    def _update_methods(self, beliefstate: BeliefState, user_acts: List[UserAct]):
        """
            Updates the method section of beliefstate to reflect what intent the user has, for
            example if the method is 'byprimarykey' the user wants information about a specific
            entity whereas if the method is 'byconstraints' the user is looking for an entity
            whose properties satisfy a set of given constraints.

            Args:
                beliefState (BeliefState): the belief state object to be updated
                user_acts (list): a list of the current user acts to be used to update the belief
                                  state
        """
        # get all different action types in user inputs
        action_types = self._get_all_usr_action_types(user_acts)
        # update methods
        # TODO: ignores act scores at the moment, have to think about a way
        # of dealing with different scores of multiple methods of the same type
        # e.g. 2 inform acts with different scores
        self._zero_all_scores(beliefstate['beliefs']['method'])
        if len(action_types) == 0:
            # no user actions
            beliefstate['beliefs']['method']['none'] = 1.0
        else:
            if UserActionType.RequestAlternatives in action_types:
                beliefstate['beliefs']['method']['byalternatives'] = 1.0
            elif UserActionType.Bye in action_types:
                beliefstate['beliefs']['method']['finished'] = 1.0
            elif UserActionType.Inform in action_types:
                # check if inform by primary key value or by constraints
                inf_by_primkey = False
                for act in user_acts:
                    if self.primary_key == act.slot:
                        inf_by_primkey = True
                if inf_by_primkey:
                    # inform by name
                    beliefstate['beliefs']['method']['byprimarykey'] = 1.0
                else:
                    # inform by constraints
                    beliefstate['beliefs']['method']['byconstraints'] = 1.0

            elif UserActionType.Request in action_types or UserActionType.Confirm in action_types:
                if UserActionType.Deny not in action_types and self._is_offer(beliefstate):
                    beliefstate['beliefs']['method']['byprimarykey'] = 1.0
            else:
                beliefstate['beliefs']['method']['none'] = 1.0


    def _is_offer(self, beliefstate: BeliefState):
        """
            Helper function for reading belief state to see if there is already a current offer;
            returns True is so, False otherwise

            Args:
                BeliefState (BeliefState): the belief state to read

            Return:
                (bool): representing if an offer has been made or not

        """
        is_offer = True
        if beliefstate['system']['lastInformedPrimKeyVal'] == '**NONE**':
            is_offer = False
        elif beliefstate['system']['lastInformedPrimKeyVal'] == '':
            is_offer = False
        return is_offer


    def _handle_user_acts(self, beliefstate: BeliefState,
                          user_acts: List[UserAct], sys_act: SysAct):

        """
            Updates the belief state based on the information contained in the user act(s)

            Args:
                beliefstate (BeliefState): the belief state to be updated
                user_act (list[UserAct]): the list of user acts to use to update the belief state
                sys_act (SysAct): the previous system act for disambiguating user acts such as
                                  affirm

        """
        gen_acts = [UserActionType.Ack,
                    UserActionType.Bad,
                    UserActionType.Hello,
                    UserActionType.Thanks]
        for act in user_acts:
            if act.type is None:
                pass
            elif act.type in gen_acts:
                slot = act.type.value
                beliefstate['beliefs']['discourseAct'][slot] = 1.0
                beliefstate['beliefs']['discourseAct']['none'] = 0.0
            elif act.type == UserActionType.Request:
                self.request_slots[act.slot] = act.score
                self.request_slots["**NONE**"] = 0.0
            elif act.type == UserActionType.Inform:
                self._handle_inform(act, beliefstate)
            elif act.type == UserActionType.Confirm:
                self._handle_confirm(act, beliefstate)
            # TODO deprecated; refactor nlu and delete
            elif act.type == UserActionType.Deny:
                self._handle_deny(act, beliefstate)
            elif act.type == UserActionType.NegativeInform:
                # reset mentioned value to zero probability
                beliefstate['beliefs'][act.slot][act.value] = 0.0
            elif act.type == UserActionType.Affirm:
                self._handle_affirm(sys_act, beliefstate)
            elif act.type in [UserActionType.Bye]:
                # nothing to do here, but needed to cirumvent warning
                pass
            elif act.type == UserActionType.RequestAlternatives:
                pass
            else:
                # unknown Dialog Act
                # To be handled:
                self.logger.warning("user act not handled by BST: " + str(act))

        self._normalize_request_scores(beliefstate)
        self._normalize_inform_scores(beliefstate)


    def _handle_inform(self, act: UserAct, beliefstate: BeliefState):
        """
            If the user gives an inform action, update the belief state to reflect the
            new information

            Args:
                act (UserAct): the user action to be parsed
                beliefstate (BeliefState): the belief state to be updated

        """
        # belief state is updated to think that slot = value
        beliefstate_slot = act.slot

        if beliefstate_slot not in self.inform_scores:
            self.inform_scores[beliefstate_slot] = {}
        self.inform_scores[beliefstate_slot][act.value] = act.score
        assert act.score == 1.0  # currently all issued actions should have probability 1.0

    def _reset_informs(self, acts : List[UserAct], beliefstate: BeliefState):
        """ Resets the inform belief for the specified (if neccessary!)

        This method is neccessary because _handle_inform can handle only 1 value per slot,
        but user could mention mutliple ones.

        Condition for reset:
            * count(acts.slot == slot) >= 1
         """

        slots = {act.slot for act in acts if act.type == UserActionType.Inform}
        for slot in slots:
            values = set() # all user-mentioned values for slot
            for act in acts:
                if act.slot == slot:
                    values.add(act.value)
            if len(values) >= 1:
                # reset, newer value received in current turn
                for value in beliefstate['beliefs'][slot]:
                    if slot in self.inform_scores:
                        self.inform_scores[slot][value] = 0.0


    def _handle_negative_inform(self, act: UserAct, beliefstate: BeliefState):
        """
            If the user gives a negative inform action, update the belief state to reflect the
            new information

            Args:
                act (UserAct): the user action to be parsed
                beliefstate (BeliefState): the belief state to be updated

        """
        if act.value is not None:
            beliefstate['beliefs'][act.slot][act.value] = 0.0
        else:
            beliefstate_slot = beliefstate['system']['lastRequestSlot']
            self.logger.dialog_turn('    DENY BST: infered slot: ' +
                                    str(beliefstate_slot))

            pass


    def _handle_affirm(self, sys_act: SysAct, beliefstate: BeliefState):
        """
            If the user gives an affirm action, update the belief state where context information
            is taken into account.

            Args:
                act (UserAct): the user action to be parsed
                beliefstate (BeliefState): the belief state to be updated

        """
        if sys_act is not None and sys_act.type == SysActionType.Confirm:
            if len(list(sys_act.slot_values.items())) == 0 \
                    or 'name' in sys_act.slot_values \
                    and sys_act.slot_values['name'] == 'none':
                pass

            beliefstate['beliefs'][list(sys_act.slot_values.keys())[0]][list(
                sys_act.slot_values.values())[0][0]] = 1.0


    def _normalize_request_scores(self, beliefstate: BeliefState):
        """
            Helper function to make sure the request scores sum up to 1 and write the scores
            to the belief state

            Args:
                beliefstate (BeliefState): the belief state to be updated
        """
        # [Request] Assigning scores equally over the requested slots that sum up 1
        if len(self.request_slots):
            sum_scores = sum([self.request_slots[slot]
                              for slot in self.request_slots])*1.0
            for slot in beliefstate['beliefs']['requested']:
                if slot in self.request_slots:
                    beliefstate['beliefs']['requested'][slot] = self.request_slots[slot]/sum_scores
                else:
                    beliefstate['beliefs']['requested'][slot] = 0.0


    def _normalize_inform_scores(self, beliefstate: BeliefState):
        """
            Helper function to make sure the inform scores for each slot sum up to 1 and write the
            scores to the belief state

            Args:
                beliefstate (BeliefState): the belief state to be updated
        """
        if len(self.inform_scores):
            for beliefstate_slot in self.inform_scores:
                sum_scores = sum([self.inform_scores[beliefstate_slot][value]
                                  for value in self.inform_scores[beliefstate_slot]])
                for value in self.inform_scores[beliefstate_slot]:
                    normalized = self.inform_scores[beliefstate_slot][value] / sum_scores
                    beliefstate['beliefs'][beliefstate_slot][value] = normalized


    def forward(self, dialog_graph, user_acts: List[UserAct] = None,
                beliefstate: BeliefState = None, sys_act: List[SysAct] = None,
                **kwargs) -> dict(beliefstate=BeliefState):

        """
            Function for updating the current dialog belief state (which tracks the system's
            knowledge about what has been said in the dialog) based on the user actions generated
            from the user's utterances

            Args:
                dialog_graph (DialogSystem): the graph to which the policy belongs
                belief_state (BeliefState): this should be None
                user_acts (list): a list of UserAct objects mapped from the user's last utterance
                sys_act (SysAct): this should be None

            Returns:
                (dict): a dictionary with the key "beliefstate" and the value the updated
                        BeliefState object

        """
        beliefstate.start_new_turn()

        if user_acts is None:
            # this check is required in case the BST is the first called module
            # e.g. usersimulation on semantic level:
            #   dialog acts as outputs -> no NLU
            return {'beliefstate': beliefstate}

        self._reset_informs(acts=user_acts, beliefstate=beliefstate)
        self._update_methods(beliefstate, user_acts)

        # TODO user acts should include probabilities and beliefstate should
        # update probabilities instead of always choosing 1.0

        # important to set these to zero since we don't want stale discourseAct
        for act in beliefstate['beliefs']['discourseAct']:
            beliefstate['beliefs']['discourseAct'][act] = 0.0
        beliefstate['beliefs']['discourseAct']['none'] = 1.0

        self.request_slots = {}

        self._handle_user_acts(beliefstate, user_acts, sys_act)

        beliefstate.update_num_dbmatches()
        return {'beliefstate': beliefstate}
