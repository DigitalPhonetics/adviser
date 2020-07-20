import os
import sys
from copy import deepcopy

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.bst import HandcraftedBST
from utils import UserActionType, UserAct


def test_initialize_bst_without_domain():
    """
    Tests whether the initialization of a BST without domain executes without errors.
    """
    HandcraftedBST()


def test_reset_beliefstate_on_start(bst):
    """
    Tests whether the beliefstate resets on dialog start.

    Args:
        bst: BST Object (given in conftest.py)
    """
    previous_bs = bst.bs
    bst.dialog_start()
    assert previous_bs is not bst.bs


def test_update_bst_without_user_act(bst):
    """
    Tests whether the BST skips an update if no user act is given.

    Args:
        bst: BST Object (given in conftest.py)
    """
    previous_bs = deepcopy(bst.bs)
    bs_dict = bst.update_bst(None)
    assert type(bs_dict) == dict
    assert len(bs_dict) == 1
    assert 'beliefstate' in bs_dict
    assert bs_dict['beliefstate'] == bst.bs
    assert str(bs_dict['beliefstate']) == str(previous_bs)


def test_update_bst_with_user_act(bst, constraintA):
    """
    Tests whether the BST updates its beliefstate if a user act is given.

    Args:
        bst: BST Object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    user_acts = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                         value=constraintA['value'])]
    previous_bs = deepcopy(bst.bs)
    bs_dict = bst.update_bst(user_acts)
    assert type(bs_dict) == dict
    assert len(bs_dict) == 1
    assert 'beliefstate' in bs_dict
    assert bs_dict['beliefstate'] == bst.bs
    assert 'user_acts' in bs_dict['beliefstate']
    assert 'num_matches' in bs_dict['beliefstate']
    assert 'discriminable' in bs_dict['beliefstate']
    assert str(bs_dict['beliefstate']) != str(previous_bs)


def test_update_bst_sets_number_of_db_matches(bst, constraintA):
    """
    Tests whether updating the BST also updates the number of matches in the database and whether they are discriminable.

    Args:
        bst: BST Object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    bst.bs['informs'][constraintA['slot']] = {constraintA['value']: 0.5}
    user_acts = [UserAct(act_type=UserActionType.Hello)]
    bs_dict = bst.update_bst(user_acts)
    assert 'num_matches' in bs_dict['beliefstate']
    assert type(bs_dict['beliefstate']['num_matches']) == int
    assert bs_dict['beliefstate']['num_matches'] > -1
    assert 'discriminable' in bs_dict['beliefstate']
    assert type(bs_dict['beliefstate']['discriminable']) == bool


def test_reset_informs_removes_slots_from_informs(bst):
    """
    Tests whether reset informs removes all given inform slots.

    Args:
        bst: BST Object (given in conftest.py)
    """
    acts = [UserAct(act_type=UserActionType.Inform, slot='foo', value='bar'),
            UserAct(act_type=UserActionType.Inform, slot='bar', value='foo')]
    bst.bs['informs'] = {'foo': {'bar': 0.5}, 'bar': {'foo': 0.3}, 'baz': {'foo': 0.6}}
    bst._reset_informs(acts)
    assert all(act.slot not in bst.bs['informs'] for act in acts)


def test_reset_informs_resets_only_informs(bst):
    """
    Tests whether reset informs removes only inform slots.

    Args:
        bst: BST Object (given in conftest.py)
    """
    acts = [UserAct(act_type=UserActionType.Inform, slot='foo', value='bar'),
            UserAct(act_type=UserActionType.Request, slot='bar', value='foo')]
    bst.bs['informs'] = {'foo': {'bar': 0.5}, 'bar': {'foo': 0.3}, 'baz': {'foo': 0.6}}
    bst._reset_informs(acts)
    assert all(act.slot not in bst.bs['informs'] for act in acts if act.type ==
               UserActionType.Inform)
    assert 'bar' in bst.bs['informs']


def test_reset_requests(bst):
    """
    Tests resetting the requests dict of the beliefstate.

    Args:
        bst: BST Object (given in conftest.py)
    """
    bst.bs['requests']['foo'] = 0.5
    bst._reset_requests()
    assert bst.bs['requests'] == {}


def test_get_all_usr_action_types(bst):
    """
    Tests whether requesting user action types will return all of them.

    Args:
        bst: BST Object (given in conftest.py)
    """
    user_action_types = [UserActionType.Inform, UserActionType.Request, UserActionType.Hello,
                         UserActionType.Thanks]
    user_acts = [UserAct(act_type=act_type) for act_type in user_action_types]
    act_type_set = bst._get_all_usr_action_types(user_acts)
    assert act_type_set == set(user_action_types)


def test_handle_user_acts_resets_informs_about_primary_key_for_inform_act(bst):
    """
    Tests whether the informs about the primary key reset when handling a user inform.

    Args:
        bst: BST Object (given in conftest.py)
    """
    bst.bs['informs'][bst.domain.get_primary_key()] = {'foo': 0.5}
    bst.bs['user_acts'] = [UserActionType.Inform]
    bst._handle_user_acts([])
    assert bst.domain.get_primary_key() not in bst.bs['informs']


def test_handle_user_acts_resets_informs_and_request_for_select_domain_act(bst):
    """
    Tests whether the informs and requests reset when handling a new domain selection by the user.

    Args:
        bst: BST Object (given in conftest.py)
    """
    bst.bs['informs']['foo'] = {'bar': 0.5}
    bst.bs['requests']['foo'] = 0.5
    bst.bs['user_acts'] = [UserActionType.SelectDomain]
    bst._handle_user_acts([])
    assert bst.bs['informs'] == {}
    assert bst.bs['requests'] == {}


def test_handle_user_acts_for_user_request(bst):
    """
    Tests whether the requests are set correctly when handling a user request.

    Args:
        bst: BST Object (given in conftest.py)
    """
    slot = 'foo'
    score = 0.5
    user_acts = [UserAct(act_type=UserActionType.Request, slot=slot, score=score)]
    bst._handle_user_acts(user_acts)
    assert slot in bst.bs['requests']
    assert bst.bs['requests'][slot] == score


def test_handle_user_acts_for_user_inform(bst):
    """
    Tests whether the informs are set correctly when handling a user inform.

    Args:
        bst: BST Object (given in conftest.py)
    """
    slot = 'foo'
    value = 'bar'
    score = 0.5
    user_acts = [UserAct(act_type=UserActionType.Inform, slot=slot, value=value, score=score)]
    bst._handle_user_acts(user_acts)
    assert slot in bst.bs['informs']
    assert value in bst.bs['informs'][slot]
    assert bst.bs['informs'][slot][value] == score


def test_handle_user_acts_for_user_negative_inform(bst):
    """
    Tests whether handling a negative inform by the user deletes the corresponding inform value.

    Args:
        bst: BST Object (given in conftest.py)
    """
    slot = 'foo'
    value = 'bar'
    bst.bs['informs'][slot] = {value: 0.5}
    user_acts = [UserAct(act_type=UserActionType.NegativeInform, slot=slot, value=value)]
    bst._handle_user_acts(user_acts)
    assert value not in bst.bs['informs'][slot]


def test_handle_user_acts_for_user_request_alternatives(bst, primkey_constraint):
    """
    Tests whether the primary key is removed from the informs when handling a user request for
    alternatives.

    Args:
        bst: BST Object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    bst.bs['informs'][primkey_constraint['slot']] = {primkey_constraint['value']: 0.5}
    user_acts = [UserAct(act_type=UserActionType.RequestAlternatives)]
    bst._handle_user_acts(user_acts)
    assert bst.domain.get_primary_key() not in bst.bs['informs']
