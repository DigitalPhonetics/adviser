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

""" This module provides the BeliefState class. """

import copy

from utils.domain.jsonlookupdomain import JSONLookupDomain


class BeliefState:
    """
    A representation of the belief state, can be accessed like a dictionary.

    Includes information on:
        * current set of UserActTypes
        * informs to date (dictionary where key is slot and value is {value: probaility})
        * requests for the current turn
        * number of db matches for given constraints
        * if the db matches can further be split

    """
    def __init__(self, domain: JSONLookupDomain):
        self.domain = domain
        self._history = [self._init_beliefstate()]
    
    def dialog_start(self):
        self._history = [self._init_beliefstate()]

    def __getitem__(self, val):  # for indexing
        # if used with numbers: int (e.g. state[-2]) or slice (e.g. state[3:6])
        if isinstance(val, int) or isinstance(val, slice):
            return self._history[val]  # interpret the number as turn
        # if used with strings (e.g. state['beliefs'])
        elif isinstance(val, str):
            # take the current turn's belief state
            return self._history[-1][val]

    def __iter__(self):
        return iter(self._history[-1])

    def __setitem__(self, key, val):
        # e.g. state['beliefs']['area']['west'] = 1.0
        self._history[-1][key] = val

    def __len__(self):
        return len(self._history)

    def __contains__(self, val):  # assume
        return val in self._history[-1]

    def _recursive_repr(self, sub_dict, indent=0):
        # if isinstance(sub_dict, type(None)):
        #     return ""
        string = ""
        if isinstance(sub_dict, dict):
            string += '{'
            for key in sub_dict:
                string += "'" + str(key) + "': "
                string += self._recursive_repr(sub_dict[key], indent + 2)
            string += '}\n' + ' ' * indent
        else:
            string += str(sub_dict) + ' '
        return string

    def __repr__(self):
        return str(self._history[-1])

    def __str__(self):
        return self._recursive_repr(self._history[-1])

    def start_new_turn(self):
        """
        ONLY to be called by the belief state tracker at the begin of each turn,
        to ensure the correct history can be accessed correctly by other modules
        """

        # copy last turn's dict
        self._history.append(copy.deepcopy(self._history[-1]))

    def _init_beliefstate(self):
        """Initializes the belief state based on the currently active domain

        Returns:
            (dict): nested dict of slots/values and system belief of
                    each state"

        """

        # TODO: revist when we include probabilites, sets should become dictionaries
        belief_state = {"user_acts": set(),
                        "informs": {},
                        "requests": {},
                        "num_matches": 0,
                        "discriminable": True}

        return belief_state

    def get_most_probable_slot_beliefs(self, slot: str, consider_NONE: bool = True,
                                       threshold: float = 0.7,
                                       max_results: int = 1, turn_idx: int = -1):
        """ Extract the most probable value for each system requestable slot

        If the most probable value for a slot does not exceed the threshold,
        then the slot will not be added to the result at all.

        Args:
            consider_NONE: If True, slots where **NONE** values have the
                           highest probability will not be added to the result.
                           If False, slots where **NONE** values have the
                           highest probability will look for the best value !=
                           **NONE**.
            threshold: minimum probability to be accepted to the
            max_results: return at most #max_results best values per slot
            turn_idx: index for accessing the belief state history (default = -1: use last turn)

        Returns:
            Union(Dict[str, List[str]], Dict[str, str]): A dict with mapping from slots to a list of values (if max_results > 1) or
            a value (if max_results == 1) containing the slots which
            have at least one value whose probability exceeds the specified
            threshold.
        """

        informs = self._history[turn_idx]["informs"]
        candidates = []
        if slot in informs:
            sorted_slot_cands = sorted(informs[slot].items(), key=lambda kv: kv[1], reverse=True)
            # restrict result count to specified maximum
            filtered_slot_cands = sorted_slot_cands[:max_results]
            # threshold by probabilities
            filtered_slot_cands = [slot_cand[0] for slot_cand in filtered_slot_cands
                                   if slot_cand[1] >= threshold]
        return candidates

    def get_most_probable_inf_beliefs(self, consider_NONE: bool = True, threshold: float = 0.7,
                                      max_results: int = 1, turn_idx: int = -1):
        """ Extract the most probable value for each system requestable slot

        If the most probable value for a slot does not exceed the threshold,
        then the slot will not be added to the result at all.

        Args:
            consider_NONE: If True, slots where **NONE** values have the
                           highest probability will not be added to the result.
                           If False, slots where **NONE** values have the
                           highest probability will look for the best value !=
                           **NONE**.
            threshold: minimum probability to be accepted
            max_results: return at most `max_results` best values per slot
            turn_idx: index for accessing the belief state history (default = -1: use last turn)

        Returns:
            Union(Dict[str, List[str]], Dict[str, str]): A dict with mapping from slots to a list of values (if max_results > 1) or
            a value (if max_results == 1) containing the slots which
            have at least one value whose probability exceeds the specified
            threshold.
        """

        candidates = {}
        informs = self._history[turn_idx]["informs"]
        for slot in informs:
            # sort by belief
            sorted_slot_cands = sorted(informs[slot].items(), key=lambda kv: kv[1], reverse=True)
            # restrict result count to specified maximum
            filtered_slot_cands = sorted_slot_cands[:max_results]
            # threshold by probabilities
            filtered_slot_cands = [slot_cand[0] for slot_cand in filtered_slot_cands
                                   if slot_cand[1] >= threshold]
            if len(filtered_slot_cands) > 0:
                # append results if any remain after filtering
                if max_results == 1:
                    # only float
                    candidates[slot] = filtered_slot_cands[0]
                else:
                    # list
                    candidates[slot] = filtered_slot_cands
        return candidates

    def get_requested_slots(self, turn_idx: int = -1):
        """ Returns the slots requested by the user

        Args:
            turn_idx: index for accessing the belief state history (default = -1: use last turn)
        """

        candidates = []
        for req_slot in self._history[turn_idx]['requests']:
            candidates.append(req_slot)
        return candidates

    def _remove_dontcare_slots(self, slot_value_dict: dict):
        """ Returns a new dictionary without the slots set to dontcare """

        return {slot: value for slot, value in slot_value_dict.items()
                if value != 'dontcare'}

    def get_num_dbmatches(self):
        """ Updates the belief state's entry for the number of database matches given the
            constraints in the current turn.
        """

        # check how many db entities match the current constraints
        candidates = self.get_most_probable_inf_beliefs(consider_NONE=True, threshold=0.7,
                                                        max_results=1)
        constraints = self._remove_dontcare_slots(candidates)
        db_matches = self.domain.find_entities(constraints, self.domain.get_informable_slots())
        num_matches = len(db_matches)

        # check if matching db entities could be discriminated by more
        # information from user
        discriminable = False
        if len(db_matches) > 1:
            dontcare_slots = set(candidates.keys()) - set(constraints.keys())
            informable_slots = set(self.domain.get_informable_slots()) - set(self.domain.get_primary_key())
            for informable_slot in informable_slots:
                if informable_slot not in dontcare_slots:
                    # this slot could be used to gather more information
                    db_values_for_slot = set()
                    for db_match in db_matches:
                        db_values_for_slot.add(db_match[informable_slot])
                    if len(db_values_for_slot) > 1:
                        # at least 2 different values for slot
                        # ->can use this slot to differentiate between entities
                        discriminable = True
                        break
        return num_matches, discriminable
