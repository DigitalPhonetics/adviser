import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from utils import SysActionType, UserActionType


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


def test_choose_sys_act_without_beliefstate(policy):
    """
    Tests whether the choose_sys_act function executes without errors when being called without
    beliefstate parameter.

    Args:
        policy: Policy Object (given in conftest.py)
    """
    execute_choose_sys_act(policy, None)


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
    #policy.first_turn = False
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
    #policy.first_turn = False
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
    #policy.first_turn = False
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
    assert sys_act['sys_act'].type == SysActionType.Request
    assert sys_act['sys_act'].slot_values == {}


def test_domain_selection_act_restarts_dialog(policy, beliefstate):
    """
    Tests whether a domain selection by the user restarts the dialog.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    #policy.first_turn = False
    beliefstate['user_acts'].add(UserActionType.SelectDomain)
    execute_choose_sys_act(policy, beliefstate)
    assert policy.turns == 0
    assert policy.current_suggestions == []
    assert policy.s_index == 0


def test_remove_filler_actions_if_one_nonfiller_actions_is_present(policy, beliefstate):
    """
    Tests whether the system ignores the filler actions if one other non-filler action was
    given by the user. Filler actions are bad / thanks/ hello actions while non-filler actions
    are for example inform or request.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].update({UserActionType.Thanks, UserActionType.Bad,
                                  UserActionType.Hello, UserActionType.Inform})
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type != SysActionType.RequestMore
    assert sys_act['sys_act'].type != SysActionType.Welcome


def test_remove_filler_actions_if_multiple_nonfiller_actions_are_present(policy, beliefstate):
    """
    Tests whether the system ignores the filler actions if multiple other non-filler actions were
    given by the user. Filler actions are bad / thanks/ hello actions while non-filler actions
    are for example inform or request.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    beliefstate['user_acts'].update({UserActionType.Thanks, UserActionType.Bad,
                                  UserActionType.Inform, UserActionType.Request})
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type != SysActionType.RequestMore
    assert sys_act['sys_act'].type != SysActionType.Welcome


@pytest.mark.parametrize('s_index', [0, 1])
def test_choose_sys_act_for_unknown_request(policy, beliefstate, s_index):
    """
    Tests the chosen system act in case of an unknown request. This can for example be triggered
    if the index for the current suggestion (s_index) is smaller or equal to the length of the
    list of current suggestions. Both of these cases are tested here.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        s_index (int): index of the current suggestion in the current suggestions list
    """
    policy.first_turn = False
    beliefstate['informs'] = {}
    beliefstate['requests'] = {'foo': 0.5}
    policy.current_suggestions = []
    policy.s_index = s_index
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Bad


def test_choose_sys_act_for_alternative_request_without_informs(policy, beliefstate):
    """
    Tests the chosen system act in case of a request for alternatives by the user but without
    any previous informs.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    #policy.first_turn = False
    beliefstate['user_acts'].add(UserActionType.RequestAlternatives)
    beliefstate['informs'] = {}
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Bad


def test_choose_sys_act_for_inform_by_primary_key(policy, beliefstate, primkey_constraint):
    """
    Tests the chosen system act in case of no specific given user act or request, but with an
    inform about
    the primary key of the domain.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    policy.first_turn = False
    beliefstate['informs'] = {primkey_constraint['slot']: {primkey_constraint['value']: 0.5}}
    beliefstate['requests'] = {}
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.InformByName
    assert policy.domain_key in sys_act['sys_act'].slot_values


def test_general_inform_about_slot_multiple_results(policy, beliefstate, constraint_with_multiple_matches):
    """
    Tests the chosen system act of no specific given user act or request, but with an inform about
    something else than the primary key of the domain, which results in multiple hits in the
    database. In case of open non-binary slots, the system will ask the user using this slot to
    further narrow down the request.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraint_with_multiple_matches (dict): slot-value pair for constraint with multiple
        matches in the domain (given in conftest_<domain>.py)
    """
    policy.first_turn = False
    beliefstate['requests'] = {}
    beliefstate['informs'] = {constraint_with_multiple_matches['slot']: {
        constraint_with_multiple_matches['value']: 0.5}}
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Request
    assert sys_act['sys_act'].slot_values != {}
    assert sys_act['sys_state']['lastRequestSlot'] != []


def test_general_inform_about_binary_slot_multiple_results(policy, beliefstate):
    """
    Tests the chosen system act of no specific given user act or request, but with an inform about
    something else than the primary key of the domain, which results in multiple hits in the
    database. In case of only open binary slots, the system will try to ask the user to narrow
    down the request by using the slot with the highest information gain.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    # TODO: cannot be tested with the superhero domain because there are no binary slots
    pass


def test_general_inform_about_binary_slot_multiple_results_and_no_diff(policy, beliefstate):
    """
    Tests the chosen system act of no specific given user act or request, but with an inform about
    something else than the primary key of the domain, which results in multiple hits in the
    database. In case of only open binary slots, the system will try to ask the user to narrow
    down the request by using the slot with the highest information gain. If there is no such
    slot, the system will return a InformByName with value "none".

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
    """
    # TODO: cannot be tested with the superhero domain because there are no binary slots
    pass


def test_general_inform_about_slot_single_result(policy, beliefstate, constraint_with_single_match):
    """
    Tests the chosen system act of no specific given user act or request, but with an inform about
    something else than the primary key of the domain, which results in a single hit in the
    database. In this case, the system will inform by its name about the entity that match the
    given inform value.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraint_with_single_match (dict): slot-value pair for constraint with a single match
        in the domain (given in conftest_<domain>.py)
    """
    policy.first_turn = False
    beliefstate['requests'] = {}
    beliefstate['informs'] = {constraint_with_single_match['slot']: {
        constraint_with_single_match['value']: 0.5}}
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.InformByName
    assert sys_act['sys_act'].slot_values != {}
    assert policy.domain_key in sys_act['sys_act'].slot_values
    assert sys_act['sys_state']['lastInformedPrimKeyVal'] == sys_act['sys_act'].slot_values[
                                                                  policy.domain_key][0]


def test_general_inform_about_slot_no_results(policy, beliefstate, constraint_with_no_match):
    """
    Tests the chosen system act of no specific given user act or request, but with an inform about
    something else than the primary key of the domain, which results in no hit in the
    database. In this case, the system will return a InformByName with value "none".

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraint_with_multiple_matches (dict): slot-value pair for constraint without matches in
        the domain (given in conftest_<domain>.py)
    """
    policy.first_turn = False
    beliefstate['requests'] = {}
    beliefstate['informs'] = {constraint_with_no_match['slot']: {constraint_with_no_match['value']: 0.5}}
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.InformByName
    assert sys_act['sys_act'].slot_values != {}
    assert sys_act['sys_act'].slot_values[policy.domain_key] == ['none']
    assert sys_act['sys_state']['lastInformedPrimKeyVal'] == 'none'


@pytest.mark.parametrize('inform', [{}, {'main_superpower': {'dontcare': 0.5}}])
def test_general_empty_inform(policy, beliefstate, inform):
    """
    Tests the chosen system act of no specific given user act or request, but with an inform that
    is empty or contains only "dontcare" values. In this case, the system will ask the user to
    narrow down the search results.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        inform (dict): 'empty' inform dicts
    """
    policy.first_turn = False
    beliefstate['requests'] = {}
    beliefstate['informs'] = inform
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.Request
    assert sys_act['sys_act'].slot_values != {}
    assert sys_act['sys_state']['lastRequestSlot'] != []


def test_inform_by_name_and_slot(policy, beliefstate, constraintA, primkey_constraint):
    """
    Tests the chosen system act of a request for something else than the primary key of the domain,
    and current suggestions containing the primary key. In this case, the system will inform about
    this specific entity by name, and will give information about the requested slot.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    policy.first_turn = False
    beliefstate['requests'] = {constraintA['slot']: 0.5}
    policy.current_suggestions = [{primkey_constraint['slot']: primkey_constraint['value']}]
    policy.s_index = 0
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.InformByName
    assert sys_act['sys_act'].slot_values != {}
    assert policy.domain_key in sys_act['sys_act'].slot_values
    assert constraintA['slot'] in sys_act['sys_act'].slot_values
    assert sys_act['sys_state']['lastInformedPrimKeyVal'] == sys_act['sys_act'].slot_values[
        policy.domain_key][0]


def test_request_alternatives_first_inform(policy, beliefstate, constraint_with_single_match):
    """
    Tests the chosen system act in case of a request for alternatives and if this is the first
    inform of the dialog. In this case, the system will inform by name about an entity.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraint_with_single_match (dict): slot-value pair for constraint with a single match
        in the domain (given in conftest_<domain>.py)
    """
    beliefstate['user_acts'].add(UserActionType.RequestAlternatives)
    beliefstate['informs'] = {constraint_with_single_match['slot']: {constraint_with_single_match['value']: 0.5}}
    policy.current_suggestions = []
    policy.s_index = 0
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.InformByName
    assert policy.current_suggestions != []
    assert policy.s_index == 0
    assert sys_act['sys_act'].slot_values != {}
    assert policy.domain_key in sys_act['sys_act'].slot_values
    assert constraint_with_single_match['slot'] in sys_act['sys_act'].slot_values
    assert sys_act['sys_state']['lastInformedPrimKeyVal'] == sys_act['sys_act'].slot_values[
        policy.domain_key][0]


def test_request_alternatives_not_first_inform(policy, beliefstate, constraint_with_single_match,
                                               primkey_constraint):
    """
    Tests the chosen system act in case of a request for alternatives and if this is not the first
    inform of the dialog. In this case, the system will inform by alternatives about an entity.

    Args:
        policy: Policy Object (given in conftest.py)
        beliefstate: BeliefState object (given in conftest.py)
        constraint_with_single_match (dict): slot-value pair for constraint with a single match
        in the domain (given in conftest_<domain>.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    beliefstate['user_acts'].add(UserActionType.RequestAlternatives)
    beliefstate['informs'] = {constraint_with_single_match['slot']: {constraint_with_single_match['value']: 0.5}}
    policy.current_suggestions = [{primkey_constraint['slot']: {primkey_constraint['value']: 0.3}}]
    policy.s_index = 0
    sys_act = execute_choose_sys_act(policy, beliefstate)
    assert sys_act['sys_act'].type == SysActionType.InformByAlternatives
    assert policy.s_index == 0
    assert sys_act['sys_act'].slot_values != {}
    assert policy.domain_key in sys_act['sys_act'].slot_values
    assert constraint_with_single_match['slot'] in sys_act['sys_act'].slot_values
    assert sys_act['sys_state']['lastInformedPrimKeyVal'] == sys_act['sys_act'].slot_values[
        policy.domain_key][0]
