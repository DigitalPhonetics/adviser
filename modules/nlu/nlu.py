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

import re
import os
import json
from typing import List

from dialogsystem import DialogSystem
from modules.module import Module
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils import UserAct, UserActionType
from utils.logger import DiasysLogger
from utils.sysact import SysAct, SysActionType
from utils.beliefstate import BeliefState
from utils.common import Language


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HandcraftedNLU(Module):
    """
    Class for Handcrafted Natural Language Understanding Module (HDC-NLU).

    HDC-NLU is a rule-based approach to recognize the user acts as well as
    their respective slots and values from the user input (i.e. text)
    by means of regular expressions.

    HDC-NLU is domain-independet. The regular expressions of are read
    from JSON files.

    There exist a JSON file that stores general rules (GeneralRules.json),
    i.e. rules that apply to any domain, e.g. rules to detect salutation (Hello, Hi).

    There are two more files per domain that contain the domain-specific rules
    for request and inform user acts, e.g. ImsCoursesInformRules.json and
    ImsCoursesRequestRules.json.

    The output during dialog interaction of this module is a semantic
    representation of the user input.

    "I am looking for pizza" --> inform(slot=food,value=italian)

    See the regex_generator under tools, if the existing regular expressions
    need to be changed or a new domain should be added.


    """

    def __init__(self, domain: JSONLookupDomain, subgraph=None,
                 logger: DiasysLogger = DiasysLogger(), language: Language = None):
        """
        Loads
            - domain key
            - informable slots
            - requestable slots
            - domain-independent regular expressions
            - domain-specific regualer espressions

        It sets the previous system act to None

        Args:
            domain {domain.jsonlookupdomain.JSONLookupDomain} -- Domain
            subgraph  {[type]} -- [see modules.Module] (default: {None})
            logger:
        """
        super(HandcraftedNLU, self).__init__(domain, None, logger=logger)

        self.language = language if language else Language.ENGLISH

        # Getting domain information
        self.domain_name = domain.get_domain_name()
        self.domain_key = domain.get_primary_key()

        # Getting lists of informable and requestable slots
        self.USER_INFORMABLE = domain.get_informable_slots()
        self.USER_REQUESTABLE = domain.get_requestable_slots()

        # Getting the relative path where regexes are stored
        self.base_folder = os.path.join(get_root_dir(), 'resources', 'regexes')


        # Setting previous system act to None to signal the first turn
        self.prev_sys_act = None

    def _match_general_act(self, user_utterance: str):
        """
        Finds general acts (e.g. Hello, Bye) in the user input

        Args:
            user_utterance {str} --  text input from user

        Returns:

        """

        # Iteration over all general acts
        for act in self.general_regex:
            # Check if the regular expression and the user utterance match
            if re.search(self.general_regex[act], user_utterance, re.I):
                # Mapping the act to User Act
                if act != 'dontcare' and act != 'req_everything':
                    user_act_type = UserActionType(act)
                else:
                    user_act_type = act
                # Check if the found user act is affirm or deny
                if self.prev_sys_act and (user_act_type == UserActionType.Affirm or
                                          user_act_type == UserActionType.Deny):
                    # Conditions to check the history in order to assign affirm or deny
                    # slots mentioned in the previous system act

                    # Check if the preceeding system act was confirm
                    if self.prev_sys_act.type == SysActionType.Confirm:
                        # Iterate over all slots in the system confimation
                        # and make a list of Affirm/Deny(slot=value)
                        # where value is taken from the previous sys act
                        for slot in self.prev_sys_act.slot_values:
                            # New user act -- Affirm/Deny(slot=value)
                            user_act = UserAct(act_type=UserActionType(act), text=user_utterance, slot=slot,
                                               value=self.prev_sys_act.slot_values[slot])
                            self.user_acts.append(user_act)

                    # Check if the preceeding system act was request
                    # This covers the binary requests, e.g. 'Is the course related to Math?'
                    elif self.prev_sys_act.type == SysActionType.Request:
                        # Iterate over all slots in the system request
                        # and make a list of Inform(slot={True|False})
                        for slot in self.prev_sys_act.slot_values:
                            # Assign value for the slot mapping from Affirm or Request to Logical,
                            # True if user affirms, False if user denies
                            value = 'true' if user_act_type == UserActionType.Affirm else 'false'
                            # Adding user inform act
                            self._add_inform(user_utterance, slot, value)

                    # Check if Deny happens after System Request more, then trigger bye
                    elif self.prev_sys_act.type == SysActionType.RequestMore \
                            and user_act_type == UserActionType.Deny:
                        user_act = UserAct(text=user_utterance, act_type=UserActionType.Bye)
                        self.user_acts.append(user_act)

                # Check if Request or Select is the previous system act
                elif user_act_type == 'dontcare':
                    if self.prev_sys_act.type == SysActionType.Request or \
                            self.prev_sys_act.type == SysActionType.Select:
                        # Iteration over all slots mentioned in the last system act
                        for slot in self.prev_sys_act.slot_values:
                            # Adding user inform act
                            self._add_inform(user_utterance, slot, value=user_act_type)

                # Check if the user wants to get all information about a particular entity
                elif user_act_type == 'req_everything':
                    self.req_everything = True

                else:
                    # This section covers all general user acts that do not depend on
                    # the dialog history
                    # New user act -- UserAct()
                    user_act = UserAct(act_type=user_act_type, text=user_utterance)
                    self.user_acts.append(user_act)

    def _match_domain_specific_act(self, user_utterance: str):
        """
        Matches in-domain user acts
        Calls functions to find user requests and informs

        Args:
            user_utterance {str} --  text input from user

        Returns:

        """
        # Find Requests
        self._match_request(user_utterance)
        # Find Informs
        self._match_inform(user_utterance)

    def _match_request(self, user_utterance: str):
        """
        Iterates over all user request regexes and find matches with the user utterance

        Args:
            user_utterance {str} --  text input from user

        Returns:

        """
        # Iteration over all user requestable slots
        for slot in self.USER_REQUESTABLE:
            if self._check(re.search(self.request_regex[slot], user_utterance, re.I)):
                self._add_request(user_utterance, slot)

    def _add_request(self, user_utterance: str, slot: str):
        """
        Creates the user request act and adds it to the user act list
        Args:
            user_utterance {str} --  text input from user
            slot {str} -- requested slot

        Returns:

        """
        # New user act -- Request(slot)
        user_act = UserAct(text=user_utterance, act_type=UserActionType.Request, slot=slot)
        self.user_acts.append(user_act)
        # Storing user requested slots during the whole dialog
        self.slots_requested.add(slot)

    def _match_inform(self, user_utterance: str):
        """
        Iterates over all user inform slot-value regexes and find matches with the user utterance

        Args:
            user_utterance {str} --  text input from user

        Returns:

        """

        # Iteration over all user informable slots and their slots
        for slot in self.USER_INFORMABLE:
            for value in self.inform_regex[slot]:
                if self._check(re.search(self.inform_regex[slot][value], user_utterance, re.I)):
                    if slot == self.domain_key and self.req_everything:
                        # Adding all requestable slots because of the req_everything
                        for req_slot in self.USER_REQUESTABLE:
                            # skipping the domain key slot
                            if req_slot != self.domain_key:
                                # Adding user request act
                                self._add_request(user_utterance, req_slot)
                    # Adding user inform act
                    self._add_inform(user_utterance, slot, value)

    def _add_inform(self, user_utterance: str, slot: str, value: str):
        """
        Creates the user request act and adds it to the user act list

        Args:
            user_utterance {str} --  text input from user
            slot {str} -- informed slot
            value {str} -- value for the informed slot

        Returns:

        """
        user_act = UserAct(text=user_utterance, act_type=UserActionType.Inform,
                           slot=slot, value=value)
        self.user_acts.append(user_act)
        # Storing user informed slots in this turn
        self.slots_informed.add(slot)

    @staticmethod
    def _exact_match(phrases: List[str], user_utterance: str) -> bool:
        """
        Checks if the user utterance is exactly like one in the

        Args:
            phrases List[str] --  list of contextual don't cares
            user_utterance {str} --  text input from user

        Returns:

        """

        # apostrophes are removed
        if user_utterance.lstrip().lower().replace("'", "") in phrases:
            return True
        return False

    def _match_affirm(self, user_utterance: str):
        """TO BE DEFINED AT A LATER POINT"""
        pass

    def _match_negative_inform(self, user_utterance: str):
        """TO BE DEFINED AT A LATER POINT"""
        pass

    @staticmethod
    def _check(re_object) -> bool:
        """
        Checks if the regular expression and the user utterance matched

        Args:
            re_object: output from re.search(...)

        Returns:
            True/False if match happened

        """

        if re_object is None:
            return False
        for o in re_object.groups():
            if o is not None:
                return True
        return False

    def _check_domain_active(self, user_utterance: str, slot: str):
        """This whole things is super sketchy, need better way of telling if a
           domain is active

           currently this checks if a regex has the correct keyword to signify
           the domain is active and if it is it returns a hello act which will
           be filtered out in the policy if any other act is detected; and
           will otherwise ask a random in domain question

        """

        # match = re.search(self.inform_regex[slot]['true'], user_utterance, re.I)
        # if self._check(match):
        #     new_act = UserAct(
        #         text=user_utterance, act_type=UserActionType.Hello)  # Temporary hack
        #     self.user_acts.append(new_act)
        pass

    def _assign_scores(self):
        """
        Goes over the user act list, checks concurrencies and assign scores

        Returns:

        """

        for i in range(len(self.user_acts)):
            # TODO: Create a clever and meaningful mechanism to assign scores
            # Since the user acts are matched, they get 1.0 as score
            self.user_acts[i].score = 1.0

    def start_dialog(self, **kwargs: dict) -> dict:
        """
        Sets the previous system act as None.
        This function is called when the dialog starts

        Args:
            **kwargs: dictionary passed through modules

        Returns:
            Empty dictionary

        """
        self.prev_sys_act = None

        return {}

    def _disambiguate_co_occurrence(self, beliefstate: BeliefState):
        # Check if there is user inform and request occur simultaneously for a binary slot
        # E.g. request(applied_nlp) & inform(applied_nlp=true)
        # Difficult to disambiguate using regexes
        if self.slots_requested.intersection(self.slots_informed):
            if beliefstate is None:
                act_to_del = UserActionType.Request
            elif beliefstate['system']['lastInformedPrimKeyVal'] in ['**NONE**', 'none']:
                act_to_del = UserActionType.Request
            else:
                act_to_del = UserActionType.Inform

            acts_to_del = []
            for slot in self.slots_requested.intersection(self.slots_informed):
                # print(self.user_acts)
                for i, user_act in enumerate(self.user_acts):
                    if user_act.type == act_to_del and user_act.slot == slot:
                        acts_to_del.append(i)

            self.user_acts = [user_act for i, user_act in enumerate(self.user_acts)
                              if i not in acts_to_del]

    def _solve_informable_values(self):
        # Verify if two or more informable slots with the same value were caught
        # Cases:
        # If a system request precedes and the slot is the on of the two informable, keep that one.
        # If there is no preceding request, take
        informed_values = {}
        for i, user_act in enumerate(self.user_acts):
            if user_act.type == UserActionType.Inform:
                if user_act.value != "true" and user_act.value != "false":
                    if user_act.value not in informed_values:
                        informed_values[user_act.value] = [(i, user_act.slot)]
                    else:
                        informed_values[user_act.value].append((i, user_act.slot))

        informed_values = {value: informed_values[value] for value in informed_values if
                           len(informed_values[value]) > 1}
        if "6" in informed_values:
            self.user_acts = []

        # # Check if there is at least one informed value that matches more than one slot
        # if len(informed_values)>0:
        #     if beliefstate['system']['lastRequestSlot'] is not None:
        #         for value in informed_values:
        #             req_slots = [i for i, slot in informed_values[value]
        #                          if beliefstate['system']['lastRequestSlot']==slot]
        #             if len(req_slots)==1:
        #                 for i, slot in informed_values[value]:
        #                     if beliefstate['system']['lastRequestSlot']!=slot:
        #                         del self.user_acts[i]
        #             # elif len(req_slots)==0:
        #             #     # it keeps the first slot matched
        #             #     # This is not the most clever solution
        #             #     for i, slot in informed_values[value][1:]:
        #             #             del self.user_acts[i]
        #     else:
        #         # If no system requested slot, then a Bad act is forced
        #         self.user_acts = []

    def _initialize(self):
        """
            Loads the correct regex files based on which language has been selected
            this should only be called on the first turn of the dialog

            Args:
                language (Language): Enum representing the language the user has selected
        """
        if self.language == Language.ENGLISH:
            # Loading regular expression from JSON files
            # as dictionaries {act:regex, ...} or {slot:{value:regex, ...}, ...}
            self.general_regex = json.load(open(self.base_folder + '/GeneralRules.json'))
            self.request_regex = json.load(open(self.base_folder + '/' + self.domain_name
                                                + 'RequestRules.json'))
            self.inform_regex = json.load(open(self.base_folder + '/' + self.domain_name
                                               + 'InformRules.json'))
        elif self.language == Language.GERMAN:
            # TODO: Change this once
            # Loading regular expression from JSON files
            # as dictionaries {act:regex, ...} or {slot:{value:regex, ...}, ...}
            self.general_regex = json.load(open(self.base_folder + '/GeneralRulesGerman.json'))
            self.request_regex = json.load(open(self.base_folder + '/' + self.domain_name
                                                + 'GermanRequestRules.json'))
            self.inform_regex = json.load(open(self.base_folder + '/' + self.domain_name
                                               + 'GermanInformRules.json'))
        else:
            print('No language')

    def _match_hard_coded_regexes(self, user_utterance: str):

        if self.language == Language.ENGLISH:
            domains = {'ImsCourses': '(lecture|course)(s)?',
                       'ImsLecturers': '(lecturer|teacher|instructor|professor)(s)?'
                       }
        elif self.language == Language.GERMAN:
            domains = {'ImsCourses': '(kurs(e)?|vorlesung(en)?)',
                       'ImsLecturers': '(dozent|lehrbeauftragter|lehrperson)(en)?'}
        regex_alterantives = "(what|which|welche) " + domains[self.domain_name]
        # print(user_utterance, regex_alterantives)

        if re.search(regex_alterantives, user_utterance, re.I):
            user_act_type = UserActionType.RequestAlternatives
            user_act = UserAct(act_type=user_act_type, text=user_utterance)
            self.user_acts.append(user_act)


    def forward(self, dialog_graph: DialogSystem, user_utterance: str = None,
        sys_act: SysAct = None, beliefstate: BeliefState = None, **kwargs: dict)\
        -> dict(user_acts=List[UserAct]):

        """
        Responsible for detecting user acts with their respective slot-values from the user
        utterance through regular expressions.

        Args:
            dialog_graph (DialogSystem) - the graph to which the policy belongs
            user_utterance (BeliefState) - a BeliefState obejct representing current system
                                           knowledge
            sys_act (list) - a list of UserAct objects mapped from the user's last utterance
            beliefstate (SysAct) - this should be None
            **kwargs - dictionary passed through modules

        Returns:
            dict of str: UserAct - a dictionary with the key "user_acts" and the value
                                            containing a list of user actions
        """

        first_turn = dialog_graph.num_turns == 0 if dialog_graph is not None else False
        if first_turn:
            self.language = Language.ENGLISH
            self._initialize()
        if "language" in kwargs and kwargs['language'] is not None:
            self.language = kwargs["language"]
            self._initialize()

        # Setting request everything to False at every turn
        self.req_everything = False

        # updating the previous system act
        if sys_act is not None:
            # If system act is Bad Act, do not update the previous system act
            if sys_act.type != SysActionType.Bad:
                self.prev_sys_act = sys_act

        self.user_acts = []

        # slots_requested & slots_informed store slots requested and informed in this turn
        # they are used later for later disambiguation
        self.slots_requested, self.slots_informed = set(), set()
        if user_utterance is not None:
            user_utterance = user_utterance.strip()
            self._match_general_act(user_utterance)
            self._match_domain_specific_act(user_utterance)
            self._match_hard_coded_regexes(user_utterance)

        # Solving ambiguities from regexes, especially with requests and informs happening
        # simultaneously on the same slot and two slots taking the same value
        self._disambiguate_co_occurrence(beliefstate)
        self._solve_informable_values()

        if len(self.user_acts) == 0 and self.prev_sys_act is not None:
            # start of dialogue or no regex matched
            self.user_acts.append(UserAct(text=user_utterance if user_utterance else "",
                                          act_type=UserActionType.Bad))

        self._assign_scores()
        self.logger.dialog_turn("User Actions: %s" % str(self.user_acts))
        return {'user_acts': self.user_acts}


if __name__ == "__main__":
    domain = JSONLookupDomain(
        'ImsCourses',
        os.path.join('resources', 'databases', 'ImsCourses-rules.json'),
        os.path.join('resources', 'databases', 'ImsCourses-dbase.db'))
    nlu = HandcraftedNLU(domain=domain)
    print(nlu.forward(None, user_utterance='tell me everything about smt')['user_acts'])
    print(len(nlu.forward(None, user_utterance='tell me everything about smt')['user_acts']))
    print(nlu.forward(None, user_utterance='vu'))
    print(nlu.forward(None, user_utterance='his time'))
    print(nlu.forward(None, user_utterance='what courses does agnieska falenska teach'))

