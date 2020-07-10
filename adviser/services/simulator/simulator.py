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

"""This module provides the agenda-based user model for the handcrafted simulator."""

import configparser
import copy
import os
from typing import List

from services.service import PublishSubscribe
from services.service import Service
from services.simulator.goal import Constraint, Goal
from utils import UserAct, UserActionType, SysAct, SysActionType, common
from utils.domain.domain import Domain
from utils.logger import DiasysLogger


class HandcraftedUserSimulator(Service):
    """The class for a handcrafted (agenda-based) user simulator.

    Args:
        domain (Domain): The domain for which the user simulator will be instantiated. It will use
        this domain to generate the goals.
    """

    def __init__(self, domain: Domain, logger: DiasysLogger = DiasysLogger()):
        super(HandcraftedUserSimulator, self).__init__(domain)

        # possible system actions
        self.receive_options = {SysActionType.Welcome: self._receive_welcome,
                                SysActionType.InformByName: self._receive_informbyname,
                                SysActionType.InformByAlternatives:
                                    self._receive_informbyalternatives,
                                SysActionType.Request: self._receive_request,
                                SysActionType.Confirm: self._receive_confirm,
                                SysActionType.Select: self._receive_select,
                                SysActionType.RequestMore: self._receive_requestmore,
                                SysActionType.Bad: self._receive_bad,
                                SysActionType.ConfirmRequest: self._receive_confirmrequest}

        # parse config file
        self.logger = logger
        self.config = configparser.ConfigParser(
            inline_comment_prefixes=('#', ';'))
        self.config.optionxform = str
        self.config.read(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'usermodel.cfg'))

        self.parameters = {}
        # goal
        self.parameters['goal'] = {}
        for key in self.config["goal"]:
            val = self.config.get("goal", key)
            self.parameters['goal'][key] = float(val)

        # usermodel
        self.parameters['usermodel'] = {}
        for key in self.config["usermodel"]:
            val = self.config.get(
                "usermodel", key)
            if key in ['patience']:
                # patience will be sampled on begin of each dialog
                self.parameters['usermodel'][key] = [int(x) for x in (
                    val.replace(' ', '').strip('[]').split(','))]
            else:
                if val.startswith("[") and val.endswith("]"):
                    # value is a list to sample the probability from
                    self.parameters['usermodel'][key] = common.numpy.random.uniform(
                        *[float(x) for x in val.replace(' ', '').strip('[]').split(',')])
                else:
                    # value is the probability
                    self.parameters['usermodel'][key] = float(val)

        # member declarations
        self.turn = 0
        self.domain = domain
        self.dialog_patience = None
        self.patience = None
        self.last_user_actions = None
        self.last_system_action = None
        self.excluded_venues = []

        # member definitions
        self.goal = Goal(domain, self.parameters['goal'])
        self.agenda = Agenda()
        self.num_actions_next_turn = -1

    def dialog_start(self):
        """Resets the user model at the beginning of a dialog, e.g. draws a new goal and populates
        the agenda according to the goal."""
        # self.goal = Goal(self.domain, self.parameters['goal'])

        self.goal.init()
        self.agenda.init(self.goal)
        if self.logger:
            self.logger.dialog_turn(
                "New goal has constraints {} and requests {}.".format(
                    self.goal.constraints, self.goal.requests))
            self.logger.dialog_turn("New agenda initialized: {}".format(self.agenda))

        # add hello action with some probability
        if common.random.random() < self.parameters['usermodel']['Greeting']:
            self.agenda.push(UserAct(act_type=UserActionType.Hello, score=1.0))

        # needed for possibility to reset patience
        if len(self.parameters['usermodel']['patience']) == 1:
            self.dialog_patience = self.parameters['usermodel']['patience'][0]
        else:
            self.dialog_patience = common.random.randint(
                *self.parameters['usermodel']['patience'])
        self.patience = self.dialog_patience
        self.last_user_actions = None
        self.last_system_action = None
        self.excluded_venues = []
        self.turn = 0

    @PublishSubscribe(sub_topics=["sys_act", "sys_turn_over"], pub_topics=["user_acts", "sim_goal"])
    def user_turn(self, sys_act: SysAct = None, sys_turn_over=False) \
            -> dict(user_acts=List[UserAct], sim_goal=Goal):
        """
        Determines the next user actions based on the given system actions and the user simulator's own goal

        Args:
            sys_act (SysAct): The system action for which a user response will be retrieved.
            sys_turn_over (bool): signal to start the user turn
        Returns:
            (dict): Dictionary including the user acts as a list and the current user's goal.

        """
        # self.turn = dialog_graph.num_turns
        if sys_act is not None and sys_act.type == SysActionType.Bye:
            # if self.goal.is_fulfilled():
            #     self._finish_dialog()
            return {"sim_goal": self.goal}

        if sys_act is not None:
            self.receive(sys_act)

        user_acts = self.respond()

        # user_acts = [UserAct(text="Hi!", act_type=UserActionType.Hello, score=1.)]

        self.logger.dialog_turn("User Action: " + str(user_acts))
        # input()
        return {'user_acts': user_acts}

    def receive(self, sys_act: SysAct):
        """
        This function makes sure that the agenda reflects all changes needed for the received
        system action.
        
        Args:
            sys_act (SysAct): The action the system took
        """

        if self.last_system_action is not None:
            # check whether system action is the same as before
            if sys_act == self.last_system_action:
                self.patience -= 1
            elif self.parameters['usermodel']['resetPatience']:
                self.patience = self.dialog_patience

        self.last_system_action = sys_act

        if self.patience == 0:
            self.logger.dialog_turn("User patience run out, ending dialog.")
            self.agenda.clear()
            self._finish_dialog(ungrateful=True)
        else:
            ignored_requests, ignored_requests_alt = self._check_system_ignored_request(
                self.last_user_actions, sys_act)

            # first stage: push operations on top of agenda
            if sys_act.type in self.receive_options:
                self.receive_options[sys_act.type](sys_act)

                # handle missing requests
                if ignored_requests:
                    # repeat unanswered requests from user from last turn
                    self.agenda.push(ignored_requests)

                if ignored_requests_alt:
                    self.agenda.push(ignored_requests_alt)
                    # make sure to pick only the requestalt actions (should be 1)
                    self.num_actions_next_turn = len(ignored_requests_alt)
                    # make sure that old request actions verifying an offer are removed
                    self.agenda.remove_actions_of_type(act_type=UserActionType.Request)

                # second stage: clean agenda
                self.agenda.clean(self.goal)
                # agenda might be empty -> add requests again
                if self.agenda.is_empty():
                    if self.goal.is_fulfilled():
                        self._finish_dialog()
                    else:
                        self.agenda.fill_with_requests(self.goal, exclude_name=False)

            else:
                self.logger.error(
                    "System Action Type is {}, but I don't know how to handle it!".format(
                        sys_act.type))

    def _receive_welcome(self, sys_act: SysAct):
        """
        Processes a welcome action from the system. In this case do nothing
        
        Args:
            sys_act (SysAct): the last system action
        """
        # do nothing as the first turn is already intercepted
        # also, the 'welcome' action is never used in reinforcement learning from the policy
        # -> will only, if at all, occur at first turn

    def _receive_informbyname(self, sys_act: SysAct):
        """
        Processes an informbyname action from the system; checks if the inform matches the
        goal constraints and if yes, will add unanswered requests to the agenda 
        
        Args:
            sys_act (SysAct): the last system action        
        """
        # check all system informs for offer
        inform_list = []
        offers = []
        for slot, value_list in sys_act.slot_values.items():
            for value in value_list:
                if slot == 'name':
                    offers.append(value)
                else:
                    inform_list.append(Constraint(slot, value))

        # check offer
        if offers:
            if self._check_offer(offers, inform_list):
                # valid offer
                for slot, value in inform_list:
                    self.goal.fulfill_request(slot, value)

        # needed to make sure that not informed constraints (which have been turned into requests)
        # will be asked first (before ending the dialog too early)
        req_actions_not_in_goal = []
        for action in self.agenda.get_actions_of_type(UserActionType.Request):
            if action.slot not in self.goal.requests:
                req_actions_not_in_goal.append(copy.deepcopy(action))

        # goal might be fulfilled now
        if (self.goal.is_fulfilled()
                and not self.agenda.contains_action_of_type(UserActionType.Inform)
                and not req_actions_not_in_goal):
            self._finish_dialog()

    def _receive_informbyalternatives(self, sys_act: SysAct):
        """
        Processes an informbyalternatives action from the system; this is treated like
        an inform by name
        
        Args:
            sys_act (SysAct): the last system action        
        """
        # same as inform by name
        if self.excluded_venues and self.goal.requests[self.domain.get_primary_key()] is None:
            self._receive_informbyname(sys_act)
        else:
            self._repeat_last_actions()

    def _receive_request(self, sys_act: SysAct):
        """
        Processes a request action from the system by adding the corresponding answer based
        on the current simulator goal.
        
        Args:
            sys_act (SysAct): the last system action        
        """
        for slot, _ in sys_act.slot_values.items():
            self.agenda.push(UserAct(
                act_type=UserActionType.Inform,
                slot=slot, value=self.goal.get_constraint(slot),
                score=1.0))

    def _receive_confirm(self, sys_act: SysAct):
        """
        Processes a confirm action from the system based on information in the user goal
        
        Args:
            sys_act (SysAct): the last system action        
        """
        for slot, _value in sys_act.slot_values.items():
            value = _value[0]  # there is always only one value
            if self.goal.is_inconsistent_constraint_strict(Constraint(slot, value)):
                # inform about correct value with some probability, otherwise deny value
                if common.random.random() < self.parameters['usermodel']['InformOnConfirm']:
                    self.agenda.push(UserAct(
                        act_type=UserActionType.Inform, slot=slot,
                        value=self.goal.get_constraint(slot),
                        score=1.0))
                else:
                    self.agenda.push(UserAct(
                        act_type=UserActionType.NegativeInform, slot=slot, value=value, score=1.0))
            else:
                # NOTE using inform currently since NLU currently does not support Affirm here and
                # NLU would tinker it into an Inform action anyway
                # self.agenda.push(
                #     UserAct(act_type=UserActionType.Affirm, score=1.0))
                self.agenda.push(
                    UserAct(act_type=UserActionType.Inform, slot=slot, value=value, score=1.0))

    def _receive_select(self, sys_act: SysAct):
        """
        Processes a select action from the system based on the simulation goal
        
        Args:
            sys_act (SysAct): the last system action        
        """
        # handle as request
        value_in_goal = False
        for slot, values in sys_act.slot_values.items():
            for value in values:
                # do not consider 'dontcare' as any value
                if not self.goal.is_inconsistent_constraint_strict(Constraint(slot, value)):
                    value_in_goal = True

        if value_in_goal:
            self._receive_request(sys_act)
        else:
            assert len(sys_act.slot_values.keys()) == 1, \
                "There shall be only one slot in a select action."
            # NOTE: currently we support only one slot for select action,
            # but this could be changed in the future
                
            slot = list(sys_act.slot_values.keys())[0]
            # inform about correct value with some probability
            if common.random.random() < self.parameters['usermodel']['InformOnSelect']:
                self.agenda.push(UserAct(
                    act_type=UserActionType.Inform, slot=slot,
                    value=self.goal.get_constraint(slot),
                    score=1.0))

            for slot, values in sys_act.slot_values.items():
                for value in values:
                    self.agenda.push(UserAct(
                        act_type=UserActionType.NegativeInform,
                        slot=slot,
                        value=value, score=1.0))

    def _receive_requestmore(self, sys_act: SysAct):
        """
        Processes a requestmore action from the system.
        
        Args:
            sys_act (SysAct): the last system action        
        """
        if self.goal.is_fulfilled():
            # end dialog
            self._finish_dialog()
        elif (not self.agenda.contains_action_of_type(UserActionType.Inform)
              and self.goal.requests['name'] is not None):
            # venue has been offered and all informs have been issued, but atleast one request slot
            # is missing
            if self.agenda.is_empty():
                self.agenda.fill_with_requests(self.goal)
        else:
            # make sure that dialog becomes longer
            self._repeat_last_actions()

    def _receive_bad(self, sys_act:SysAct):
        """
        Processes a bad action from the system; repeats the last user action
        
        Args:
            sys_act (SysAct): the last system action        
        """
        # NOTE repeat last action, should never occur on intention-level as long no noise is used
        self._repeat_last_actions()

    def _receive_confirmrequest(self, sys_act: SysAct):
        """
        Processes a confirmrequest action from the system.
        
        Args:
            sys_act (SysAct): the last system action        
        """
        # first slot is confirm, second slot is request
        for slot, value in sys_act.slot_values.items():
            if value is None:
                # system's request action
                self._receive_request(
                    SysAct(act_type=SysActionType.Request, slot_values={slot: None}))
            else:
                # system's confirm action
                # NOTE SysActionType Confirm has single value only
                self._receive_confirm(
                    SysAct(act_type=SysActionType.Confirm, slot_values={slot: [value]}))

    def respond(self):
        """
        Gets n actions from the agenda, where n is drawn depending on the agenda or a pdf.
        """
        # get some actions from the agenda

        assert len(self.agenda) > 0, "Agenda is empty, this must not happen at this point!"

        if self.num_actions_next_turn > 0:
            # use and reset self.num_actions_next_turn if set
            num_actions = self.num_actions_next_turn
            self.num_actions_next_turn = -1
        elif self.agenda.stack[-1].type == UserActionType.Bye:
            # pop all actions from agenda since agenda can only contain thanks (optional) and
            # bye action
            num_actions = -1
        else:
            # draw amount of actions
            num_actions = min(len(self.agenda), common.numpy.random.choice(
                [1, 2, 3], p=[.6, .3, .1]))  # hardcoded pdf

        # get actions from agenda
        user_actions = self.agenda.get_actions(num_actions)
        # copy needed for repeat action since they might be changed in other modules
        self.last_user_actions = copy.deepcopy(user_actions)

        for action in user_actions:
            if action.type == UserActionType.Inform:
                _constraint = Constraint(action.slot, action.value)
                # if _constraint in self.goal.constraints:
                if action in self.goal.missing_informs:
                    self.goal.missing_informs.remove(action)

        return user_actions

    def _finish_dialog(self, ungrateful=False):
        """
        Pushes a bye action ontop of the agenda in order to end a dialog. Depending on the user
        model, a thankyou action might be added too.

        Args:
            ungrateful (bool): determines if the user should also say "thanks"; if the dialog ran
                               too long or the user ran out of patience, ungrateful will be true
        """
        self.agenda.clear()  # empty agenda
        # thank with some probability
        # NOTE bye has to be the topmost action on the agenda since we check for it in the
        # respond() method
        if not ungrateful and common.random.random() < self.parameters['usermodel']['Thank']:
            self.agenda.push(UserAct(act_type=UserActionType.Thanks, score=1.0))
        self.agenda.push(UserAct(act_type=UserActionType.Bye, score=1.0))

    def _repeat_last_actions(self):
        """
        Pushes the last user actions ontop of the agenda.
        """
        if self.last_user_actions is not None:
            self.agenda.push(self.last_user_actions[::-1])
            self.num_actions_next_turn = len(self.last_user_actions)

    def _alter_constraints(self, constraints, count):
        """
        Alters *count* constraints from the given constraints by choosing a new value
        (could be also 'dontcare').
        """
        constraints_candidates = constraints[:]  # copy list
        if not constraints_candidates:
            for _constraint in self.goal.constraints:
                if _constraint.value != 'dontcare':
                    constraints_candidates.append(Constraint(_constraint.slot, _constraint.value))
        else:
            # any constraint from the current system actions has to be taken into consideration
            # make sure that constraints are part of the goal since noise could have influenced the
            # dialog -> given constraints must conform to the current goal
            constraints_candidates = list(filter(
                lambda x: not self.goal.is_inconsistent_constraint_strict(x),
                constraints_candidates))

        if not constraints_candidates:
            return []

        constraints_to_alter = common.numpy.random.choice(
            constraints_candidates, count, replace=False)

        new_constraints = []
        for _constraint in constraints_to_alter:
            self.goal.excluded_inf_slot_values[_constraint.slot].add(
                _constraint.value)
            possible_values = self.goal.inf_slot_values[_constraint.slot][:]
            for _value in self.goal.excluded_inf_slot_values[_constraint.slot]:
                # remove values which have been tried already
                # NOTE values in self.excluded_inf_slot_values should always be in possible_values
                # because the same source is used for both and to initialize the goal
                possible_values.remove(_value)

            if not possible_values:
                # add 'dontcare' as last option
                possible_values.append('dontcare')

            # 'dontcare' value with some probability
            if common.random.random() < self.parameters['usermodel']['DontcareIfNoVenue']:
                value = 'dontcare'
            else:
                value = common.numpy.random.choice(possible_values)
            if not self.goal.update_constraint(_constraint.slot, value):
                # NOTE: this case should never happen!
                print(
                    "The given constraints (probably by the system) are not part of the goal!")
            new_constraints.append(Constraint(_constraint.slot, value))

        self.logger.dialog_turn(
            "Goal altered! {} -> {}.".format(constraints_to_alter, new_constraints))

        return new_constraints

    def _check_informs(self, informed_constraints_by_system):
        """ Checks whether the informs by the system are consistent with the goal and pushes
        appropriate actions onto the agenda for inconsistent constraints. """

        # check for inconsistent constraints and remove informs of consistent constraints from
        # agenda
        consistent_with_goal = True
        for _constraint in informed_constraints_by_system:
            if self.goal.is_inconsistent_constraint(_constraint):
                consistent_with_goal = False
                self.agenda.push(UserAct(
                    act_type=UserActionType.Inform,
                    slot=_constraint.slot,
                    value=self.goal.get_constraint(_constraint.slot), score=1.0))
            else:
                self.agenda.remove_actions(UserActionType.Inform, *_constraint)

        return consistent_with_goal

    def _check_offer(self, offers, informed_constraints_by_system):
        """ Checks for an offer and returns True if the offer is valid. """

        if not self._check_informs(informed_constraints_by_system):
            # reset offer in goal since inconsistencies have been detected and covered
            self.goal.requests[self.domain.get_primary_key()] = None
            return False

        # TODO maybe check for current offer first since alternative with name='none' by system
        # would trigger goal change -> what is the correct action in this case?
        if offers:
            if 'none' not in offers:
                # offer was given

                # convert informs of values != 'dontcare' to requests
                actions_to_convert = list(self.agenda.get_actions_of_type(
                    UserActionType.Inform, consider_dontcare=False))
                if len(self.goal.constraints) > 1 and len(actions_to_convert) == len(self.goal.constraints):
                    # penalise too early offers
                    self._repeat_last_actions()
                    self.num_actions_next_turn = len(self.last_user_actions)
                    return False

                # ask for values of remaining inform slots on agenda - this has two purposes:
                #   1. making sure that offer is consistent with goal
                #   2. making sure that inconsistent offers prolongate a dialog
                for action in actions_to_convert:
                    self.agenda.push(UserAct(
                        act_type=UserActionType.Request,
                        slot=action.slot,
                        value=None, score=1.0))
                self.agenda.remove_actions_of_type(UserActionType.Inform)

                if self.goal.requests[self.domain.get_primary_key()] is not None:
                    if self.goal.requests[self.domain.get_primary_key()] in offers:
                        # offer is the same, don't change anything but treat offer as valid
                        return True
                    else:
                        # offer is not the same, but did not request a new one
                        # NOTE with current bst do not (negative) inform about the offer, because
                        # it will only set the proability to zero -> will not be excluded
                        # self.agenda.push(UserAct(act_type=UserActionType.NegativeInform,\
                        #     slot=self.domain.get_primary_key(), value=offers[0]))
                        return False
                else:
                    for _offer in offers:
                        if _offer not in self.excluded_venues:
                            # offer is not on the exclusion list (e.g. from reqalt action) and
                            # there is no current offer

                            # sometimes ask for alternative
                            if common.random.random() < self.parameters['usermodel']['ReqAlt']:
                                self._request_alt(_offer)
                                return False
                            else:
                                self.goal.requests[self.domain.get_primary_key()] = _offer
                                for _action in self.goal.missing_informs:
                                    # informed constraints by system are definitely consistent with
                                    # goal at this point
                                    if Constraint(_action.slot, _action.value) not in informed_constraints_by_system:
                                        self.agenda.push(UserAct(
                                            act_type=UserActionType.Request,
                                            slot=_action.slot,
                                            value=None))
                                return True

                    # no valid offer was given
                    self._request_alt()
                    return False
            else:
                # no offer was given
                # TODO add probability to choose number of alternations
                altered_constraints = self._alter_constraints(informed_constraints_by_system, 1)
                # reset goal push new actions on top of agenda
                self.goal.reset()
                self.goal.missing_informs = [UserAct(
                    act_type=UserActionType.Inform,
                    slot=_constraint.slot,
                    value=_constraint.value) for _constraint in self.goal.constraints]
                for _constraint in altered_constraints:
                    self.agenda.push(UserAct(
                        act_type=UserActionType.Inform,
                        slot=_constraint.slot,
                        value=_constraint.value,
                        score=1.0))
                self.agenda.clean(self.goal)
                return False
        return False

    def _request_alt(self, offer=None):
        """
        Handles the case where a user might want to ask for an alternative offer
        """
        # add current offer to exclusion list, reset current offer and request alternative
        if offer is not None:
            self.excluded_venues.append(offer)
        if self.goal.requests[self.domain.get_primary_key()] is not None:
            self.excluded_venues.append(self.goal.requests[self.domain.get_primary_key()])
            self.goal.requests[self.domain.get_primary_key()] = None
        self.goal.reset()
        self.agenda.push(UserAct(act_type=UserActionType.RequestAlternatives))

    def _check_system_ignored_request(self, user_actions: List[UserAct], sys_act: SysAct):
        """
        Make sure that there are no unanswered requests/constraints that got turned into requests
        """
        if not user_actions:
            # no user_actions -> system ignored nothing
            return [], []

        requests = [action for action in user_actions if action.type == UserActionType.Request]
        
        if not requests:
            # no requests -> system ignored nothing
            return [], []
        
        if sys_act.type in [SysActionType.InformByName]:
            requests = [request for request in requests if request.slot not in sys_act.slot_values]

        requests_alt = [action for action in user_actions if action.type == UserActionType.RequestAlternatives]
        if sys_act.type == SysActionType.InformByAlternatives:
            offer = sys_act.slot_values[self.domain.get_primary_key()]

            if (set(offer) - set(self.excluded_venues)):  # and self.goal.requests[self.domain.get_primary_key()] is None:
                requests_alt = []

        return requests, requests_alt


class Agenda(object):
    """
    A stack-like object representing an agenda. Actions can be pushed on and popped off the agenda.
    """

    def __init__(self):
        self.stack = []

    def __iter__(self):
        return iter(self.stack)

    def __contains__(self, value):
        return value in self.stack

    def __len__(self):
        return len(self.stack)

    def __bool__(self):
        return bool(self.stack)

    def __repr__(self):
        return repr(self.stack)

    def __str__(self):
        return str(self.stack)

    def init(self, goal):
        """
        Initializes the agenda given a goal. For this purpose, inform actions for constraints in
        the goal and request actions for requests in the goal are added such that the informs are
        handled first followed by the requests.

        Args:
            goal (Goal): The goal for which the agenda will be initialized.

        """
        self.stack.clear()
        # populate agenda according to goal

        # NOTE don't push bye action here since bye action could be poppped with another (missing)
        # request, but user should not end dialog before having the goal fulfilled

        # NOTE do not add requests to agenda since system can't handle inform and request action in
        # same turn currently!
        # self.fill_with_requests(goal)

        self.fill_with_constraints(goal)

    def push(self, item):
        """Pushes *item* onto the agenda.

        Args:
            item: The goal for which the agenda will be initialized.

        """
        if isinstance(item, list):
            self.stack += item
        else:
            self.stack.append(item)

    def get_actions(self, num_actions: int):
        """Retrieves *num_actions* actions from the agenda.

        Args:
            num_actions (int): Amount of actions which will be retrieved from the agenda.

        Returns:
            (List[UserAct]): list of *num_actions* user actions.

        """

        if num_actions < 0 or num_actions > len(self.stack):
            num_actions = len(self.stack)

        return [self.stack.pop() for _ in range(num_actions)]

    def clean(self, goal: Goal):
        """Cleans the agenda, i.e. makes sure that actions are consistent with goal and in the
        correct order.

        Args:
            goal (Goal): The goal which is needed to determine the consistent actions.

        """
        cleaned_stack = []
        # reverse order since most recent actions are on top of agenda
        for action in self.stack[::-1]:
            if action not in cleaned_stack:
                # NOTE sufficient if there is only one slot per (request) action
                # remove accomplished requests
                if (action.type is not UserActionType.Request
                        or (action.slot in goal.requests and goal.requests[action.slot] is None)
                        or action.slot not in goal.requests):
                    # make sure to remove "old" inform actions
                    if action.type is UserActionType.Inform:
                        if not goal.is_inconsistent_constraint(
                                Constraint(action.slot, action.value)):
                            cleaned_stack.insert(0, action)
                    else:
                        cleaned_stack.insert(0, action)
        self.stack = cleaned_stack

    def clear(self):
        """Empties the agenda."""
        self.stack.clear()

    def is_empty(self):
        """Checks whether the agenda is empty.

        Returns:
            (bool): True if agenda is empty, False otherwise.

        """
        return len(self.stack) == 0

    def contains_action_of_type(self, act_type: UserActionType, consider_dontcare=True):
        """Checks whether agenda contains actions of a specific type.

        Args:
            act_type (UserActionType): The action type (intent) for which the agenda will be checked.
            consider_dontcare (bool): If set to True also considers actions for which the value is
                                     'dontcare', and ignores them otherwise.

        Returns:
            (bool): True if agenda contains *act_type*, False otherwise.

        """
        for _action in self.stack:
            if not consider_dontcare and _action.value == 'dontcare':
                continue
            if _action.type == act_type:
                return True
        return False

    def get_actions_of_type(self, act_type: UserActionType, consider_dontcare: bool = True):
        """Get actions of a specific type from the agenda.

        Args:
            act_type (UserActionType): The action type (intent) for which the agenda will be checked.
            consider_dontcare (bool): If set to True also considers actions for which the value is
                                     'dontcare', and ignores them otherwise.
        Returns:
            (Iterable[UserAct]): A list of user actions of the given type/intent.

        """
        return filter(
            lambda x: x.type == act_type
                      and (consider_dontcare or x.value != 'dontcare'), self.stack)

    def remove_actions_of_type(self, act_type: UserActionType):
        """Removes actions of a specific type from the agenda.

        Args:
            act_type (UserActionType): The action type (intent) which will be removed from the agenda.

        """
        self.stack = list(filter(lambda x: x.type != act_type, self.stack))

    def remove_actions(self, act_type: UserActionType, slot: str, value: str = None):
        """Removes actions of a specific type, slot and optionally value from the agenda. All
        arguments (value only if given) have to match in conjunction.

        Args:
            act_type (UserActionType): The action type (intent) which will be removed from the agenda.
            slot (str): The action type (intent) which will be removed from the agenda.
            value (str): The action type (intent) which will be removed from the agenda.

        """
        if value is None:
            self.stack = list(filter(lambda x: x.type != act_type or x.slot != slot, self.stack))
        else:
            self.stack = list(filter(
                lambda x: x.type != act_type or x.slot != slot or x.value != value, self.stack))

    def fill_with_requests(self, goal: Goal, exclude_name: bool = True):
        """Adds all request actions to the agenda necessary to fulfill the *goal*.

        Args:
            goal (Goal): The current goal of the (simulated) user for which actions will be pushed to the
                         agenda.
            exclude_name (bool): whehter or not to include an action to request an entities name.

        """
        # add requests and make sure to add the name at the end (i.e. ask first for name)
        for key, value in goal.requests.items():
            if ((key != 'name' and exclude_name) or not exclude_name) and value is None:
                self.stack.append(
                    UserAct(act_type=UserActionType.Request, slot=key, value=value, score=1.0))

    def fill_with_constraints(self, goal: Goal):
        """
        Adds all inform actions to the agenda necessary to fulfill the *goal*. Generally there is
        no need to add all constraints from the goal to the agenda apart from the initialisation.

        Args:
            goal (Goal): The current goal of the (simulated) user for which actions will be pushed to the agenda.

        """

        # add informs from goal
        for constraint in goal.constraints:
            self.stack.append(UserAct(
                act_type=UserActionType.Inform,
                slot=constraint.slot,
                value=constraint.value, score=1.0))
