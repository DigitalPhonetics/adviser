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

"""This module provides the Goal class and related stuff."""

import copy

from utils import common, UserAct, UserActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain


class Constraint(object):
    def __init__(self, slot, value):
        """
        The class for a constraint as used in the goal.

        Args:
            slot (str): The slot.
            value (str): The value.

        """
        self.slot = slot
        self.value = value

    def __eq__(self, other):
        """Constraint should be equal if the slot and value is the same."""
        if isinstance(other, Constraint):
            return (self.slot == other.slot
                    and self.value == other.value)
        return False

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key == 0:
            return self.slot
        elif key == 1:
            return self.value
        else:
            raise IndexError

    def __repr__(self):
        return "Constraint(slot={}, value={})".format(self.slot, self.value)

    def __hash__(self):
        return hash(self.slot) * hash(self.value)


class Goal(object):
    def __init__(self, domain: JSONLookupDomain, parameters=None):
        """
        The class representing a goal, therefore containing requests and constraints.

        Args:
            domain (JSONLookupDomain): The domain for which the goal will be instantiated.
                It will only work within this domain.
            parameters (dict): The parameters for the goal defined by a key=value mapping: 'MinVenues'
                (int) allows to set a minimum number of venues which fulfill the constraints of the goal,
                'MinConstraints' (int) and 'MaxConstraints' (int) set the minimum and maximum amount of
                constraints respectively, 'MinRequests' (int) and 'MaxRequests' (int) set the minimum and
                maximum amount of requests respectively and 'Reachable' (float) allows to specify how many
                (in percent) of all generated goals are definitely fulfillable (i.e. there exists a venue
                for the current goal) or not (doesn't have to be fulfillable). Although the parameter
                'Reachable' equals 1.0 implicitly states that 'MinVenues' equals 1 or more, the
                implementation looks different, is more efficient and takes all goals into consideration
                (since 'Reachable' is a float (percentage of generated goals)). On the other hand, setting
                'MinVenues' to any number bigger than 0 forces every goal to be fulfillable.

        """
        self.domain = domain
        self.parameters = parameters or {}

        # cache inform and request slots
        # make sure to copy the list (shallow is sufficient)
        self.inf_slots = sorted(list(domain.get_informable_slots())[:])
        # make sure that primary key is never a constraint
        if self.domain.get_primary_key() in self.inf_slots:
            self.inf_slots.remove(self.domain.get_primary_key())

        # TODO sometimes ask for specific primary key with very small probability (instead of any other constraints?) # pylint: disable=line-too-long

        self.inf_slot_values = {}
        for slot in self.inf_slots:
            self.inf_slot_values[slot] = sorted(
                domain.get_possible_values(slot)[:])
        self.req_slots = sorted(domain.get_requestable_slots()[:])
        # self.req_slots_without_informables = sorted(list(
        #     set(self.req_slots).difference(self.inf_slots)))
        # make sure that primary key is never a request as it is added anyway
        if self.domain.get_primary_key() in self.req_slots:
            self.req_slots.remove(self.domain.get_primary_key())

        self.constraints = []
        self.requests = {}
        self.excluded_inf_slot_values = {}
        self.missing_informs = []

    def init(self, random_goal=True, constraints=None, requests=None) -> None:
        """
        Initializes a goal randomly OR using the given constraints and requests.

        Args:
            random_goal (bool): If True, a goal will be drawn randomly from available constraints
                and requests (considering the parameters given in the constructor, if any). However if
                constraints and requests are given and both don't equal None, this parameter is
                considered as False. If False, the given constraints and requests are used.
            constraints (List[Constraint]): The constraints which will be used for the goal.
            requests (Dict[str, Union[None,str]]): The requests which will be used for the goal.

        """
        # reset goal
        self.constraints = []
        self.requests = {}
        self.excluded_inf_slot_values = {key: set()
                                         for key in self.inf_slot_values}

        # TODO implement possibility to pass either constraints or requests as a parameter
        if random_goal and constraints is None and requests is None:
            self._init_random_goal()
        else:
            self._init_from_parameters(constraints, requests)

        # make sure that primary key is always requested
        self.requests[self.domain.get_primary_key()] = None

        self.missing_informs = [UserAct(act_type=UserActionType.Inform, slot=_constraint.slot, value=_constraint.value)
                                for _constraint in self.constraints]

    def _init_random_goal(self):
        """Randomly sets the constraints and requests for the goal."""
        num_venues = -1
        # check that there exist at least self.parameters['MinVenues'] venues for this goal
        if 'MinVenues' in self.parameters:
            min_venues = self.parameters['MinVenues']
        else:
            min_venues = 0  # default is to not have any lower bound
        while num_venues < min_venues:
            # TODO exclude 'dontcare' from goal
            # TODO make sure that minconstraints and minrequests are a valid number for the current domain # pylint: disable=line-too-long

            if 'MaxConstraints' in self.parameters:
                num_constraints_max = min(len(self.inf_slots), int(
                    self.parameters['MaxConstraints']))
            else:
                # NOTE could become pretty high
                num_constraints_max = len(self.inf_slots)
            if 'MinConstraints' in self.parameters:
                num_constraints_min = int(self.parameters['MinConstraints'])
            else:
                num_constraints_min = 1

            # draw constraints uniformly
            num_constraints = common.random.randint(
                num_constraints_min, num_constraints_max)
            constraint_slots = common.numpy.random.choice(
                self.inf_slots, num_constraints, replace=False)

            self.constraints = []
            if ('Reachable' in self.parameters
                    and common.random.random() < self.parameters['Reachable']):
                # pick entity from database and set constraints
                results = self.domain.find_entities(
                    constraints={}, requested_slots=constraint_slots.tolist())
                assert results, "Cannot receive entity from database,\
                        probably because the database is empty."
                entity = common.random.choice(results)
                for constraint in constraint_slots:
                    self.constraints.append(Constraint(
                        constraint, entity[constraint]))
            else:
                # pick random constraints
                for constraint in constraint_slots:
                    self.constraints.append(Constraint(constraint, common.numpy.random.choice(
                        self.inf_slot_values[constraint], size=1)[0]))

            # check if there are enough venues for the current goal
            num_venues = len(self.domain.find_entities(constraints={
                constraint.slot: constraint.value for constraint in self.constraints}))

            possible_req_slots = sorted(
                list(set(self.req_slots).difference(constraint_slots)))
            if 'MaxRequests' in self.parameters:
                num_requests_max = min(len(possible_req_slots), int(
                    self.parameters['MaxRequests']))
            else:
                # NOTE could become pretty high
                num_requests_max = len(possible_req_slots)
            if 'MinRequests' in self.parameters:
                num_requests_min = int(self.parameters['MinRequests'])
            else:
                num_requests_min = 0  # primary key is included anyway

            num_requests = common.random.randint(
                num_requests_min, num_requests_max)
            self.requests = {slot: None for slot in common.numpy.random.choice(
                possible_req_slots, num_requests, replace=False)}
            # print(self.requests)
            # print(self.constraints)
            # TODO add some remaining informable slot as request with some probability
            # add_req_slots_candidates = list(set(self.inf_slots).difference(constraint_slots))

    def _init_from_parameters(self, constraints, requests):
        """Converts the given constraints and requests to the goal."""
        # initialise goal with given constraints and requests
        if isinstance(constraints, list):
            if constraints:
                if isinstance(constraints[0], Constraint):
                    self.constraints = copy.deepcopy(constraints)
                if isinstance(constraints[0], tuple):
                    # assume tuples in list with strings (slot, value)
                    self.constraints = [Constraint(
                        slot, value) for slot, value in constraints]
        elif isinstance(constraints, dict):
            self.constraints = [Constraint(slot, value)
                                for slot, value in constraints.items()]
        else:
            raise ValueError(
                "Given constraints for goal must be of type list or dict.")

        if not isinstance(requests, dict):
            if isinstance(requests, list):
                # assume list of strings
                self.requests = dict.fromkeys(requests, None)
        else:
            self.requests = requests

        num_venues = len(self.domain.find_entities(constraints={
            constraint.slot: constraint.value for constraint in self.constraints}))
        if 'MinVenues' in self.parameters:
            assert num_venues >= self.parameters['MinVenues'], "There are not enough venues for\
                the given constraints in the database. Either change constraints or lower\
                parameter Goal:MinVenues."

    def reset(self):
        """Resets all requests of the goal."""
        # reset goal -> empty all requests
        self.requests = dict.fromkeys(self.requests)
        # for slot in self.requests:
        #     self.requests[slot] = None

    def __repr__(self):
        return "Goal(constraints={}, requests={})".format(self.constraints, self.requests)

    # NOTE only checks if requests are fulfilled,
    # not whether the result from the system conforms to the constraints
    def is_fulfilled(self):
        """
        Checks whether all requests have been fulfilled.

        Returns:
            bool: ``True`` if all requests have been fulfilled, ``False`` otherwise.

        .. note:: Does not check whether the venue (issued by the system) fulfills the constraints
            since it's the system's task to give an appropriate venue by requesting the user's
            constraints.

        """
        for slot, value in self.requests.items():
            assert slot != self.domain.get_primary_key() or value != 'none'  # TODO remove later
            if value is None:
                return False
        return True

    def fulfill_request(self, slot, value):
        """
        Fulfills a request, i.e. sets ``value`` for request ``slot``.

        Args:
            slot (str): The request slot which will be filled.
            value (str): The value the request slot will be filled with.

        """
        if slot in self.requests:
            self.requests[slot] = value

    # does not consider 'dontcare's
    # NOTE better use is_inconsistent_constraint or is_inconsistent_constraint_strict
    # def contains_constraint(self, constraint):
    #     if constraint in self.constraints:
    #         return True
    #     return False

    # constraint is consistent with goal if values match or value in goal is 'dontcare'
    def is_inconsistent_constraint(self, constraint):
        """
        Checks whether the given constraint is consistent with the goal. A constraint is also
        consistent if it's value is 'dontcare' in the current goal.

        Args:
            constraint (Constraint): The constraint which will be checked for consistency.

        Returns:
            bool: True if values match or value in goal is 'dontcare', False otherwise.

        """
        for _constraint in self.constraints:
            if _constraint.slot == constraint.slot and (_constraint.value != constraint.value \
                                                        and _constraint.value != 'dontcare'):
                return True
        return False

    # constraint is consistent with goal if values match
    # ('dontcare' is considered as different value)
    def is_inconsistent_constraint_strict(self, constraint):
        """
        Checks whether the given constraint is strictly consistent with the goal, whereby
        'dontcare' is treated as a different value (no match).

        Args:
            constraint (Constraint): The constraint which will be checked for consistency.

        Returns:
            bool: True if values match, False otherwise.

        !!! seealso "See Also"
            [`is_inconsistent_constraint`][adviser.services.simulator.goal.Goal.is_inconsistent_constraint]

        """
        for _constraint in self.constraints:
            if _constraint.slot == constraint.slot and _constraint.value == constraint.value:
                return False
        # here there are only two possibilities: the constraint is implicitly 'dontcare' because
        # it is not explicitly listed and the given constraint is either 1) 'dontcare' or 2) not
        return constraint.value != 'dontcare'

    def get_constraint(self, slot):
        """
        Gets the value for a given constraint ``slot``.

        Args:
            slot (str): The constraint ``slot`` which will be looked up.

        Returns:
            bool: The constraint ``value``.

        """
        for _constraint in self.constraints:
            if _constraint.slot == slot:
                return _constraint.value
        return 'dontcare'

    # update the constraint with the slot 'slot' with 'value', assuming the constraints are unique
    def update_constraint(self, slot, value):
        """
        Update a given constraint ``slot`` with ``value``.

        Args:
            slot (str): The constraint *slot* which will be updated.
            value (str): The *value* with which the constraint will be updated.

        Returns:
            bool: ``True`` if update was successful, i.e. the constraint ``slot`` is included in
            the goal, ``False`` otherwise.

        """
        for _constraint in self.constraints:
            if _constraint.slot == slot:
                _constraint.value = value
                return True
        return False
