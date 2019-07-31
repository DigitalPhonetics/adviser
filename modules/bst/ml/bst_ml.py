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

import os

import torch.nn.functional as F
import spacy

from modules.module import Module
from utils.beliefstate import BeliefState
from utils.useract import UserActionType, UserAct
from utils import SysAct, SysActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain
from typing import List
from utils.logger import DiasysLogger
from modules.bst.ml.dstc_data import DSTC2Data
import modules.bst.ml.dstc_data as DSTC
from modules.bst.ml.belief_tracker import InformableTracker, RequestableTracker


glove_embedding_dim = 300
word_lvl_gru_hidden_dim = 100
dense_output_dim = 50

try:
    nlp = spacy.load('en_core_web_sm')
    DSTC._load_glove_embedding(glove_embedding_dim, 
                        data_path=os.path.join(os.path.realpath(os.curdir),
                                                "modules", "bst", "ml" ))
except:
    pass


class MLBST(Module):
    """
    A machine learning based approach on belief state tracking for 
    informables and requestables.
    Methods and discourse-Acts are evaluated by rules.
    The state is basically a dictionary of keys.
    """

    def __init__(self, domain : JSONLookupDomain = None, subgraph=None, 
                 logger : DiasysLogger =  DiasysLogger()):
        super(MLBST, self).__init__(domain, subgraph, logger = logger)
        self.path_to_data_folder = os.path.join(os.path.realpath(os.curdir),
                                            "modules", "bst", "ml"
                                    )
        self.primary_key = domain.get_domain_name()

        self.data_mappings = DSTC2Data(path_to_data_folder=self.path_to_data_folder,
                                       preprocess=False, load_train_data=False)
    
        self.inf_trackers = {}
        self.req_trackers = {}

        for inf_slot in domain.get_informable_slots():
            self._load_inf_model(inf_slot)
        for req_slot in domain.get_requestable_slots():
            self._load_req_model(req_slot)

    def _load_inf_model(self, slot_name):
        if slot_name == self.domain.get_primary_key():
            return
        slot_val_count = self.data_mappings.count_informable_slot_values(slot_name)
        self.inf_trackers[slot_name] = InformableTracker(self.data_mappings.get_vocabulary(), 
                                    slot_name, slot_val_count, glove_embedding_dim,
                                    word_lvl_gru_hidden_dim, dense_output_dim=50,
                                    p_dropout=0.5)
        self.inf_trackers[slot_name].load(model_path=self.path_to_data_folder)
        self.inf_trackers[slot_name].eval()


    def _load_req_model(self, slot_name):
        self.req_trackers[slot_name] = RequestableTracker(self.data_mappings.get_vocabulary(), 
                                    slot_name, glove_embedding_dim,
                                    word_lvl_gru_hidden_dim, dense_output_dim=50,
                                    p_dropout=0.5)
        self.req_trackers[slot_name].load(model_path=self.path_to_data_folder)
        self.req_trackers[slot_name].eval()

    def start_dialog(self, **kwargs):
        # initialize belief state
       return {'beliefstate': BeliefState(self.domain)}

    def _zero_all_scores(self, slot_values):
        """ Sets all scores of the slot-value dict to 0.0 """
        for slot in slot_values:
            slot_values[slot] = 0.0

    def _get_all_usr_action_types(self, user_acts):
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

    def _sysact_to_list(self, act, list_to_append):
        list_to_append.append(act.type.value)
        for slot in act.slot_values:
            if not 'infom' in act.type.value and not 'select' in act.type.value:
                for val in act.slot_values[slot]:
                    list_to_append.append(slot)
                    list_to_append.append(val)

    def forward(self, dialog_graph, user_utterance: str = "", 
                user_acts: List[UserAct] = None,
                beliefstate: BeliefState = None, sys_act: List[SysAct] = None,
                **kwargs) -> dict(beliefstate=BeliefState):
        # initialize belief state
        if isinstance(beliefstate, type(None)):
            beliefstate = BeliefState(self.domain)
        else:
            beliefstate.start_new_turn()

        if user_acts is None:
            # this check is required in case the BST is the first called module
            # e.g. usersimulation on semantic level:
            #   dialog acts as outputs -> no NLU
            return {'beliefstate': beliefstate}

        # get all different action types in user inputs
        action_types = self._get_all_usr_action_types(user_acts)
        # update methods
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
                inform_primkeyval = \
                    [primkey_inform_act for primkey_inform_act in user_acts
                     if primkey_inform_act.type == UserActionType.Inform and
                        self.primary_key == primkey_inform_act.slot]
                if len(inform_primkeyval) > 0:
                    # inform by name
                    beliefstate['beliefs']['method']['byprimarykey'] = 1.0
                else:
                    # inform by constraints
                    beliefstate['beliefs']['method']['byconstraints'] = 1.0
            elif (UserActionType.Request in action_types or
                  UserActionType.Confirm in action_types) and \
                 not UserActionType.Inform in action_types and \
                 not UserActionType.Deny in action_types and \
                 not (beliefstate['system']['lastInformedPrimKeyVal'] == '**NONE**' or
                      beliefstate['system']['lastInformedPrimKeyVal'] == ''):
                    beliefstate['beliefs']['method']['byprimarykey'] = 1.0
            else:
                beliefstate['beliefs']['method']['none'] = 1.0

        # important to set these to zero since we don't want stale discourseAct
        for act in beliefstate['beliefs']['discourseAct']:
            beliefstate['beliefs']['discourseAct'][act] = 0.0
        beliefstate['beliefs']['discourseAct']['none'] = 1.0

        for act_in in user_acts:
            if act_in.type is None:
                pass
            elif act_in.type is UserActionType.Bad:
                beliefstate['beliefs']['discourseAct']['none'] = 0.0
                beliefstate['beliefs']['discourseAct']['bad'] = 1.0
            elif act_in.type == UserActionType.Hello:
                beliefstate['beliefs']['discourseAct']['none'] = 0.0
                beliefstate['beliefs']['discourseAct']['hello'] = 1.0
            elif act_in.type == UserActionType.Thanks:
                beliefstate['beliefs']['discourseAct']['none'] = 0.0
                beliefstate['beliefs']['discourseAct']['thanks'] = 1.0
            # TODO adapt user actions to dstc action names
            elif act_in.type in [UserActionType.Bye]:
                # nothing to do here, but needed to cirumwent warning
                pass
            elif act.type == UserActionType.RequestAlternatives:
                pass
            elif act_in.type == UserActionType.Ack:
                beliefstate['beliefs']['discourseAct']['none'] = 0.0
                beliefstate['beliefs']['discourseAct']['ack'] = 1.0
            # else:
            #     # unknown Dialog Act
            #     # To be handled:
            #     self.logger.warning("user act not handled by BST: " + str(act_in))
                
        # track informables and requestables
        utterance = " ".join([tok.text for tok in nlp(user_utterance.strip().lower())]) # tokenize
        # convert system utterance to text triples
        sys_utterance = []
        self._sysact_to_list(sys_act, sys_utterance)
        sys_utterance = " ".join(sys_utterance)

        # track informables
        for key in self.inf_trackers:
            output = self.inf_trackers[key].forward(sys_utterance, utterance, first_turn=(dialog_graph.num_turns==0))
            output = output.squeeze()
            probabilities = F.softmax(output, dim=0)
            # print(probabilities)
            # top_value = self.data_mappings.get_informable_slot_value(key, probabilities.argmax(0).item())
            # top_prob = probabilities.max(0)[0].item()

            # copy beliefstate from network to complete beliefstate for policy
            for val in self.data_mappings.inf_values[key]:
                beliefstate['beliefs'][key][val] = probabilities[self.data_mappings.get_informable_slot_value_index(key, val)].item()
            
            # print("slot ", key)
            # print("    most probable:", top_value, " with prob", top_prob)

        for key in self.req_trackers:
            output = self.req_trackers[key].forward(sys_utterance, utterance, first_turn=(dialog_graph.num_turns==0))
            output = output.squeeze()
            probabilities = F.softmax(output, dim=0)
            # print(probabilities)
            # top_value = bool(probabilities.argmax(0).item())
            # top_prob = probabilities.max(0)[0].item()
            # print("req slot ", key)
            # print("    most probable:", top_value, " with prob", top_prob) 
              # copy beliefstate from network to complete beliefstate for policy
            beliefstate['beliefs']['requested'][key] = probabilities[1].item() # true 
            

        return {'beliefstate': beliefstate}
