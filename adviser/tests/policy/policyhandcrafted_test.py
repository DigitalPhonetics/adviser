import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from utils import SysAct, SysActionType, UserActionType


def execute_choose_sys_act(policy, beliefstate):
    """
    Tests the return value of the choose_sys_act method and makes sure that the return value is
    always a dictionary of a certain structure. Should be called in each test handling
    choose_sys_act to make sure that it returns in every case the same structure.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)

    Returns:
        (dict): a dictionary containing the system's next action
    """
    sys_act = policy.choose_sys_act(beliefstate)
    assert type(sys_act) == dict
    assert len(sys_act) == 2
    assert sys_act['sys_act']
    assert sys_act['sys_state']
    assert sys_act['sys_state']['last_act'] == sys_act['sys_act']
    return sys_act


# Tests

def test_reset_turn_count_on_start(policy):
    """
    Tests whether the turn count resets to zero when calling dialog_start.

    Args:
        policy: Policy Object (given in conftest.py)
    """
    policy.turns = 12
    policy.dialog_start()
    assert policy.turns == 0
    assert policy.first_turn is True


def test_reset_suggestions_on_start(policy):
    """
    Tests whether the suggestions list resets to an empty list when calling dialog_start.

    Args:
        policy: Policy Object (given in conftest.py)
    """
    policy.current_suggestions = ['foo', 'ba']
    policy.dialog_start()
    assert policy.current_suggestions == []
    assert policy.s_index == 0


def test_choose_sys_act_in_first_turn(policy, beliefstate):
    """
    Tests the chosen system act in the first turn when no user act was provided.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)

    """
    policy.first_turn = True
    beliefstate['user_acts'] = set()
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Welcome
    assert policy.first_turn == False


def test_choose_sys_act_in_first_turn_with_user_acts(policy, beliefstate):
    """
    Tests the chosen system act in the first turn when a user act was provided.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)

    """
    policy.first_turn = True
    beliefstate['user_acts'].add(UserActionType.Bye)
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type != SysActionType.Welcome
    assert policy.first_turn == False


@pytest.mark.parametrize('turns', [100, 10])
def test_choose_sys_act_in_last_turn(policy, beliefstate, turns):
    """
    Tests the chosen system act in a turn that exceeds the number of maximal turns or equals it.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        turns (int): number of system's turns
    """
    policy.first_turn = False
    policy.turns = turns
    policy.max_turns = 10
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Bye


def test_choose_sys_act_for_bad_act(policy, beliefstate):
    """
    Tests the chosen system act in case of a bad act by the user.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].add(UserActionType.Bad)
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Bad


def test_choose_sys_act_for_bye_act(policy, beliefstate):
    """
    Tests the chosen system act in case of a bye act by the user.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].add(UserActionType.Bye)
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Bye


def test_choose_sys_act_for_thanks_act(policy, beliefstate):
    """
    Tests the chosen system act in case of a thanks act by the user.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].add(UserActionType.Thanks)
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.RequestMore


@pytest.mark.parametrize('user_act_type', [UserActionType.Hello, UserActionType.SelectDomain])
def test_choose_sys_act_with_empty_informs_slots(policy, beliefstate, user_act_type):
    """
    Tests the chosen system act in case of a hello or domain selection act by the user if no
    informs slots are given.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        user_act_type (UserActionType): A UserActionType the system should be tested with.
    """
    beliefstate['user_acts'].add(user_act_type)
    beliefstate['informs'] = {}
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Request
    assert policy.first_turn == False
    assert sys_act['sys_act'].slot_values != {}


@pytest.mark.parametrize('user_act_type', [UserActionType.Hello, UserActionType.SelectDomain])
def test_choose_sys_act_with_open_informs_slots(policy, beliefstate, user_act_type):
    """
    Tests the chosen system act in case of a hello or domain selection act by the user if informs
    slots are given which do not cover all possible slots. This means that there are other system
    requestable slots for the domain that the system did not inform about yet.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        user_act_type (UserActionType): A UserActionType the system should be tested with.
    """
    beliefstate['user_acts'].add(UserActionType.Hello)
    slot = policy.domain.get_system_requestable_slots()[0]
    beliefstate['informs'] = {slot: {'foo': 0.5}}
    sys_act = execute_choose_sys_act(policy, beliefstate) # slot
    assert sys_act['sys_act'].type == SysActionType.Request
    assert sys_act['sys_act'].slot_values != {}


@pytest.mark.parametrize('user_act_type', [UserActionType.Hello, UserActionType.SelectDomain])
def test_choose_sys_act_without_open_informs_slots(policy, beliefstate, user_act_type):
    """
    Tests the chosen system act in case of a hello or domain selection act by the user if informs
    slots are given which cover all possible slots. This means that there are no other system
    requestable slots for the domain that the system did not inform about yet.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        user_act_type (UserActionType): A UserActionType the system should be tested with.
    """
    beliefstate['user_acts'].add(UserActionType.Hello)
    all_slots = policy.domain.get_system_requestable_slots()
    beliefstate['informs'] = {slot: {'foo': 0.5} for slot in all_slots}
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.RequestMore
    assert sys_act['sys_act'].slot_values == {}


def test_domain_selection_act_restarts_dialog(policy, beliefstate):
    """
    Tests whether a domain selection by the user restarts the dialog.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].add(UserActionType.SelectDomain)
    execute_choose_sys_act(policy, beliefstate)
    assert policy.turns == 0
    assert policy.current_suggestions == []
    assert policy.s_index == 0


def test_remove_filler_actions_if_one_nonfiller_actions_is_present(policy, beliefstate):
    """
    Tests whether the system removes the filler actions if one other non-filler action was
    given by the user. Filler actions are bad / thanks/ hello actions while non-filler actions
    are for example inform or request.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].update({UserActionType.Thanks, UserActionType.Bad,
                                  UserActionType.Hello, UserActionType.Inform})
    policy._remove_gen_actions(beliefstate)
    assert UserActionType.Thanks not in beliefstate['user_acts']
    assert UserActionType.Bad not in beliefstate['user_acts']
    assert UserActionType.Hello not in beliefstate['user_acts']


def test_remove_gen_actions_if_multiple_nonfiller_actions_are_present(policy, beliefstate):
    """
    Tests whether the system removes the filler actions if multiple other non-filler actions were
    given by the user. Filler actions are bad / thanks/ hello actions while non-filler actions
    are for example inform or request.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].update({UserActionType.Thanks, UserActionType.Bad,
                                  UserActionType.Inform, UserActionType.Request})
    policy._remove_gen_actions(beliefstate)
    assert UserActionType.Thanks not in beliefstate['user_acts']
    assert UserActionType.Bad not in beliefstate['user_acts']


def test_remove_gen_actions_removes_only_gen_actions(policy, beliefstate):
    """
    Tests whether the system only removes filler actions but not non-filler actions. Filler actions are bad / thanks/ hello actions while non-filler actions
    are for example inform or request.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    all_actions = set([act_type for act_type in UserActionType])
    gen_actions = {UserActionType.Thanks, UserActionType.Bad, UserActionType.Hello}
    non_gen_actions = all_actions - gen_actions
    beliefstate['user_acts'] = set(all_actions)
    policy._remove_gen_actions(beliefstate)
    assert beliefstate['user_acts'] == non_gen_actions


def test_query_db_with_name_and_requests(policy, beliefstate, primkey_constraint, constraintA):
    """
    Tests whether querying the database with a given name and requested slots returns information
    about these slots for the given entity.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {primkey_constraint['slot']: {primkey_constraint['value']: 0.5}}
    beliefstate['requests'] = {constraintA['slot']: 0.5}
    res = policy._query_db(beliefstate)
    assert len(res) == 1
    assert constraintA['slot'] in res[0]


def test_query_db_without_name_or_requests(policy, beliefstate, constraintA):
    """
    Tests whether querying the database without a given name or requested slots returns
    information about all entities that satisfy the constraints given in the informs dictionary.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {constraintA['slot']: {constraintA['value']: 0.5}}
    beliefstate['requests'] = {}
    res = policy._query_db(beliefstate)
    assert len(res) > 0
    assert all(constraintA['slot'] in item for item in res)


def test_get_name_with_primary_key_in_informs(policy, beliefstate, primkey_constraint):
    """
    Tests whether a name given in the informs dictionary is recognized as such.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {primkey_constraint['slot']: {primkey_constraint['value']: 0.5}}
    res = policy._get_name(beliefstate)
    assert res is not None
    assert res == primkey_constraint['value']


def test_get_name_with_suggested_name(policy, beliefstate, primkey_constraint):
    """
    Tests whether an entity name is found in the list of current suggestions if no name is
    specified in the informs dictionary.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {}
    policy.current_suggestions = [{primkey_constraint['slot']: primkey_constraint['value']}]
    policy.s_index = 0
    res = policy._get_name(beliefstate)
    assert res is not None
    assert res == primkey_constraint['value']


def test_get_name_without_name(policy, beliefstate):
    """
    Tests whether no name is found if it is not given in the informs dictionary or in the list of
    current suggestions.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['informs'] = {}
    policy.current_suggestions = []
    policy.s_index = 0
    res = policy._get_name(beliefstate)
    assert res is None


def test_get_constraints_with_constraints(policy, beliefstate, constraintA):
    """
    Tests whether retrieving given constraints returns a list of constraint slots.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
#         conftest_<domain>.py)
    """
    beliefstate['informs'] = {constraintA['slot']: {constraintA['value']: 0.5}}
    slots, _ = policy._get_constraints(beliefstate)
    assert len(slots) > 0
    assert all(value != 'dontcare' for value in slots.keys())


def test_get_constraints_without_constraints(policy, beliefstate):
    """
    Tests whether trying to retrieve constraints without given constraints returns no slots.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['informs'] = {}
    slots, _ = policy._get_constraints(beliefstate)
    assert len(slots) == 0


def test_get_constraints_with_dontcare_constraints(policy, beliefstate, constraintA):
    """
    Tests whether retrieving constraints which have a 'dontcare' value returns these as dontcare
    slots.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {constraintA['slot']: {'dontcare': 0.5}}
    _, dontcare = policy._get_constraints(beliefstate)
    assert len(dontcare) > 0


def test_get_constraints_without_dontcare_constraints(policy, beliefstate, constraintA):
    """
    Tests whether retrieving constraints which do not have 'dontcare' values returns no dontcare
    slots.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {constraintA['slot']: {constraintA['value']: 0.5}}
    _, dontcare = policy._get_constraints(beliefstate)
    assert len(dontcare) == 0


def test_get_open_slot_with_open_slot(policy, beliefstate):
    """
    Tests whether retrieving open slots will a return system requestable open slot if there is one.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    system_requestable_slots = policy.domain.get_system_requestable_slots()
    if system_requestable_slots:
        beliefstate['informs'] = {'foo': {'bar': 0.5}}
        res = policy._get_open_slot(beliefstate)
        assert res is not None



def test_get_open_slot_without_open_slots(policy, beliefstate):
    """
    Tests whether retrieving open slots will return nothing if the user has already specified
    constraints for all system requestable slots

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    system_requestable_slots = policy.domain.get_system_requestable_slots()
    beliefstate['informs'] = {slot: {'foo': 0.5} for slot in system_requestable_slots}
    res = policy._get_open_slot(beliefstate)
    assert res is None


def test_next_action_for_bad_user_action(policy, beliefstate):
    """
    Tests whether the system selects a bad action if the user performed a bad action.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'] = {UserActionType.Bad}
    sys_act, res = policy._next_action(beliefstate)
    assert sys_act.type == SysActionType.Bad
    assert type(res) == dict
    assert res == {'last_act': sys_act}


def test_next_action_for_request_with_unknown_name(policy, beliefstate, constraintA):
    """
    Tests whether the system selects a bad action if the user performed a request for something
    for which the name is not known.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {}
    beliefstate['requests'] = {constraintA['slot']: 0.5}
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act, res = policy._next_action(beliefstate)
    assert sys_act.type == SysActionType.Bad
    assert type(res) == dict
    assert res == {'last_act': sys_act}


def test_next_action_for_alternative_request_without_informs(policy, beliefstate):
    """
    Tests whether the system selects a bad action if the user performed a request for
    alternatives
    but did not give any informs.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].add(UserActionType.RequestAlternatives)
    beliefstate['informs'] = {}
    sys_act, res = policy._next_action(beliefstate)
    assert sys_act.type == SysActionType.Bad
    assert type(res) == dict
    assert res == {'last_act': sys_act}


def test_next_action_for_inform_by_primary_key(policy, beliefstate, primkey_constraint):
    """
    Tests whether the system selects an inform by name action if the user performed an inform
    about the primary key of the domain but did not request anything.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {primkey_constraint['slot']: {primkey_constraint['value']: 0.5}}
    beliefstate['requests'] = {}
    sys_act, res = policy._next_action(beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert primkey_constraint['slot'] in sys_act.slot_values
    assert primkey_constraint['value'] in sys_act.slot_values[primkey_constraint['slot']]
    assert type(res) == dict
    assert res == {'last_act': sys_act}


def test_next_action_with_multiple_matches(policy, beliefstate, constraint_with_multiple_matches):
    """
    Tests whether the system selects a request action if several entities match the given
    constraints.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraint_with_multiple_matches (dict): slot-value pair for constraint with multiple
        matches in the domain (given in conftest_<domain>.py)
    """
    beliefstate['requests'] = {}
    beliefstate['informs'] = {constraint_with_multiple_matches['slot']: {
        constraint_with_multiple_matches['value']: 0.5}}
    sys_act, sys_state = policy._next_action(beliefstate)
    assert sys_act.type == SysActionType.Request
    assert sys_state['last_act'] == sys_act
    assert sys_state['lastRequestSlot'] is not None


def test_next_action_with_single_match(policy, beliefstate, constraint_with_single_match):
    """
    Tests whether the system selects an inform by name action if only one entity matches the
    given constraints.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraint_with_single_match (dict): slot-value pair for constraint with a single match
        in the domain (given in conftest_<domain>.py)
    """
    primkey = policy.domain.get_primary_key()
    beliefstate['requests'] = {}
    beliefstate['informs'] = {constraint_with_single_match['slot']: {
        constraint_with_single_match['value']: 0.5}}
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act, sys_state = policy._next_action(beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert sys_state['last_act'] == sys_act
    assert primkey in sys_act.slot_values
    assert sys_state['lastInformedPrimKeyVal'] == sys_act.slot_values[primkey][0]


def test_next_action_with_no_match(policy, beliefstate, constraint_with_no_match):
    """
    Tests whether the system selects an inform by name action about an entity with name 'none' if no entities match the given constraints.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraint_with_no_match (dict): slot-value pair for constraint without matches in
        the domain (given in conftest_<domain>.py)
    """
    primkey = policy.domain.get_primary_key()
    beliefstate['requests'] = {}
    beliefstate['informs'] = {constraint_with_no_match['slot']: {constraint_with_no_match['value']: 0.5}}
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act, sys_state = policy._next_action(beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert sys_act.slot_values[primkey] == ['none']
    assert sys_state['lastInformedPrimKeyVal'] == 'none'


def test_raw_action_for_one_result(policy, beliefstate, entryA):
    """
    Tests whether the system selects an inform by name action in case of only one given query
    result.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    q_res = [entryA]
    sys_act = policy._raw_action(q_res, beliefstate)
    assert sys_act.type == SysActionType.InformByName


def test_raw_action_for_given_request(policy, beliefstate, entryA, entryB, constraintA):
    """
    Tests whether the system selects an inform by name action in case of several query results but
     with given requests.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
        entryB (dict): slot-value pairs for another complete entry in the domain (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    q_res = [entryA, entryB]
    beliefstate['requests'] = {constraintA['slot']: 0.5}
    sys_act = policy._raw_action(q_res, beliefstate)
    assert sys_act.type == SysActionType.InformByName


def test_raw_action_for_multiple_results(policy, beliefstate, entryA, entryB):
    """
    Tests whether the system selects a request action in case of several query results and
    without given requests.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
        entryB (dict): slot-value pairs for another complete entry in the domain (given in
        conftest_<domain>.py)
    """
    q_res = [entryA, entryB]
    beliefstate['requests'] = {}
    sys_act = policy._raw_action(q_res, beliefstate)
    assert sys_act.type == SysActionType.Request
    assert sys_act.slot_values != {}


def test_gen_next_request_with_non_binary_slots(policy, beliefstate):
    """
    Tests whether the system returns a slot for which multiple non-binary values were retrieved
    from the database if such a slot exists.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    system_requestable = [slot for slot in policy.domain.get_system_requestable_slots() if
                          len(policy.domain.get_possible_values(slot)) != 2]
    if system_requestable:
        temp = {system_requestable[0]: ['foo', 'bar']}
        temp.update({slot: ['foo'] for slot in system_requestable[1:]})
        beliefstate['informs'] = {}
        slot = policy._gen_next_request(temp, beliefstate)
        assert slot != ""
        assert slot == system_requestable[0]


def test_gen_next_request_with_non_binary_slot_with_one_value(policy, beliefstate):
    """
     Tests whether the system returns no slot if for all non-binary slots at most one value and
     no binary slot were retrieved from the database.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    system_requestable = policy.domain.get_system_requestable_slots()
    temp = {slot: ['foo'] if len(policy.domain.get_possible_values(slot)) != 2  else []
            for slot in system_requestable}
    beliefstate['informs'] = {}
    slot = policy._gen_next_request(temp, beliefstate)
    assert slot == ""


def test_gen_next_request_with_binary_slots(policy, beliefstate):
    """
     Tests whether the system returns a binary slot if for no non-binary slots but only binary
     slots multiple values were retrieved from the database.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    system_requestable = policy.domain.get_system_requestable_slots()
    binary_slots = [slot for slot in system_requestable if
                   len(policy.domain.get_possible_values(slot)) == 2]
    if binary_slots:
        bin_slot = binary_slots[0]
        val1, val2 = policy.domain.get_possible_values(bin_slot)
        temp = {bin_slot: [val1] * 3 + [val2] * 4}
        temp.update({slot: [] for slot in system_requestable if slot != bin_slot})
        beliefstate['informs'] = {}
        slot = policy._gen_next_request(temp, beliefstate)
        assert slot != ""
        assert slot == bin_slot


def test_highest_info_gain_with_difference_between_slots(policy, constraint_binaryA,
                                                         constraint_binaryB):
    """
    Tests whether the binary slot with the highest information gain is selected if there are
    binary slots for which both possible values have been retrieved from the database.

    Args:
        policy: Policy Object (given in conftest.py)
        constraint_binaryA (dict): an existing slot-value pair in the domain for a slot with only
        two possible values (given in conftest_<domain>.py)
        constraint_binaryB (dict): another existing slot-value pair in the domain for a slot with
        only two possible values (given in conftest_<domain>.py)
    """
    if constraint_binaryA and constraint_binaryB:
        val1A, val2A = policy.domain.get_possible_values(constraint_binaryA['slot'])
        val1B, val2B = policy.domain.get_possible_values(constraint_binaryB['slot'])
        temp = {
            constraint_binaryA['slot']: [val1A] * 3 + [val2A] * 4,
            constraint_binaryB['slot']: [val1B] + [val2B] * 5
        }
        bin_slots = [constraint_binaryA['slot'], constraint_binaryB['slot']]
        slot = policy._highest_info_gain(bin_slots, temp)
        assert slot != ""
        assert slot == constraint_binaryA['slot']


def test_highest_info_gain_without_difference_between_slots(policy, constraint_binaryA,
constraint_binaryB):
    """
    Tests whether no slot is selected if there is no binary slot for which more than one of the
    two possible values have been retrieved from the database.

    Args:
        policy: Policy Object (given in conftest.py)
        constraint_binaryA (dict): an existing slot-value pair in the domain for a slot with only
        two possible values (given in conftest_<domain>.py)
        constraint_binaryB (dict): another existing slot-value pair in the domain for a slot with
        only two possible values (given in conftest_<domain>.py)
    """
    if constraint_binaryA and constraint_binaryB:
        temp = {
            constraint_binaryA['slot']: [constraint_binaryA['value']] * 3 ,
            constraint_binaryB['slot']: [constraint_binaryB['value']]
        }
        bin_slots = [constraint_binaryA['slot'], constraint_binaryB['slot']]
        slot = policy._highest_info_gain(bin_slots, temp)
        assert slot == ""


def test_convert_inform_with_primary_key(policy, beliefstate, primkey_constraint, entryA):
    """
    Tests whether the system selects an inform by name action if the inform is converted with a
    given primary key.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {primkey_constraint['slot']: {primkey_constraint['value']: 0.5}}
    sys_act = SysAct()
    policy._convert_inform([entryA], sys_act, beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert sys_act.slot_values != {}


def test_convert_inform_for_request_alternatives(policy, beliefstate, entryA):
    """
    Tests whether the system selects either an inform by name action or an inform by alternatives
    action if the inform is converted with a given request for alternatives by the user.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['user_acts'] = [UserActionType.RequestAlternatives]
    beliefstate['requests'] = []
    sys_act = SysAct()
    policy._convert_inform([entryA], sys_act, beliefstate)
    assert sys_act.type in (SysActionType.InformByName, SysActionType.InformByAlternatives)
    assert sys_act.slot_values != {}


def test_convert_inform_with_given_constraints(policy, beliefstate, entryA):
    """
    Tests whether the system selects an inform by name action if the inform is converted with
    other constraints than the primary key or a request for alternatives.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['user_acts'] = []
    beliefstate['requests'] = []
    sys_act = SysAct()
    policy._convert_inform([entryA], sys_act, beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert sys_act.slot_values != {}


def test_convert_inform_by_primkey_with_results(policy, beliefstate, entryA):
    """
    Tests whether the system converts the inform system action to an inform by name action with
    a specified name if querying the database returned results.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    primkey = policy.domain.get_primary_key()
    beliefstate['informs'] = {primkey: {entryA[primkey]: 0.5}}
    sys_act = SysAct()
    policy._convert_inform_by_primkey([entryA], sys_act, beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert primkey in sys_act.slot_values
    assert entryA[primkey] in sys_act.slot_values[primkey]


def test_convert_inform_by_primkey_without_results(policy, beliefstate):
    """
    Tests whether the system converts the inform system action to an inform by name action with
    no specified name if querying the database returned no results.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    primkey = policy.domain.get_primary_key()
    sys_act = SysAct()
    policy._convert_inform_by_primkey([], sys_act, beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert primkey in sys_act.slot_values
    assert 'none' in sys_act.slot_values[primkey]


def test_convert_inform_by_alternatives_adds_constraints(policy, beliefstate, entryA,
                                                         constraintA, constraintB):
    """
    Tests whether converting the inform system action by a request for alternatives adds the
    constraints as slot-value pairs to the system action.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    beliefstate['informs'] = {
        constraintA['slot']: {constraintA['value']: 0.5},
        constraintB['slot']: {constraintB['value']: 0.3}
    }
    sys_act = SysAct()
    policy._convert_inform_by_alternatives(sys_act, [entryA], beliefstate)
    assert any(slot == constraintA['slot'] and constraintA['value'] in values for slot, values in
               sys_act.slot_values.items())
    assert any(slot == constraintB['slot'] and constraintB['value'] in values for slot, values in
               sys_act.slot_values.items())


def test_convert_inform_by_alternatives_without_suggestions(policy, beliefstate, entryA):
    """
    Tests whether the system converts the inform system action to an inform by name action if it
    is the first inform of the dialog (i.e. the list of current suggestions is empty).

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act = SysAct()
    policy._convert_inform_by_alternatives(sys_act, [entryA], beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert policy.current_suggestions != []
    assert policy.s_index == 0
    assert sys_act.slot_values != {}


def test_convert_inform_by_alternatives_with_suggestions(policy, beliefstate, entryA, entryB):
    """
    Tests whether the system converts the inform system action to an inform by alternatives action
    if it is not the first inform of the dialog (i.e. the list of current suggestions is not empty).

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
        entryB (dict): slot-value pairs for another complete entry in the domain (given in
        conftest_<domain>.py)
    """
    primkey = policy.domain.get_primary_key()
    policy.current_suggestions = [entryA, entryB]
    policy.s_index = 0
    sys_act = SysAct()
    policy._convert_inform_by_alternatives(sys_act, [], beliefstate)
    assert sys_act.type == SysActionType.InformByAlternatives
    assert policy.s_index == 1
    assert sys_act.slot_values != {}
    assert entryB[primkey] in sys_act.slot_values[primkey]


def test_convert_inform_by_alternatives_with_invalid_index(policy, beliefstate, entryA, entryB):
    """
    Tests whether the system converts the inform system action to an inform by name action
    without a specific entity name if the index for the current suggestion list is not valid
    (e.g. the index is higher than the number of current suggestions).

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
        entryB (dict): slot-value pairs for another complete entry in the domain (given in
        conftest_<domain>.py)
    """
    primkey = policy.domain.get_primary_key()
    policy.current_suggestions = [entryA]
    policy.s_index = 1
    sys_act = SysAct()
    policy._convert_inform_by_alternatives(sys_act, [entryB], beliefstate)
    assert sys_act.type == SysActionType.InformByAlternatives
    assert policy.s_index == 0
    assert sys_act.slot_values != {}
    assert 'none' in sys_act.slot_values[primkey]


def test_convert_inform_by_constraints_with_results(policy, beliefstate, entryA, entryB):
    """
    Tests whether the system converts the inform system action to an inform by name action with
    specific entity name if querying the database returned results.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
        entryB (dict): slot-value pairs for another complete entry in the domain (given in
        conftest_<domain>.py)
    """
    primkey = policy.domain.get_primary_key()
    policy.current_suggestions = [entryA]
    policy.s_index = 1
    sys_act = SysAct()
    policy._convert_inform_by_constraints([entryB], sys_act, beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert policy.current_suggestions == [entryB]
    assert policy.s_index == 0
    assert entryB[primkey] in sys_act.slot_values[primkey]


def test_convert_inform_by_constraints_without_results(policy, beliefstate):
    """
    Tests whether the system converts the inform system action to an inform by name action with
    no specific entity name if querying the database returned no results.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    primkey = policy.domain.get_primary_key()
    sys_act = SysAct()
    policy._convert_inform_by_constraints([], sys_act, beliefstate)
    assert sys_act.type == SysActionType.InformByName
    assert 'none' in sys_act.slot_values[primkey]