import os
import sys

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.bst import HandcraftedBST
from utils import UserActionType, UserAct



def execute_update_bst(bst, user_acts):
    """
    Tests the return value of the update_bst method and makes sure that the return value is
    always a dictionary with the key beliefstate. Should be called in each test handling
    update_bst to make sure that it returns in every case the same structure.

    Args:
        bst: BST Object (given in conftest.py)
        user_acts (list): List of user acts

    Returns:
        (dict): a dictionary containing the beliefstate
    """
    bs_dict = bst.update_bst(user_acts)
    assert type(bs_dict) == dict
    assert len(bs_dict) == 1
    assert 'beliefstate' in bs_dict
    assert bs_dict['beliefstate'] is not None
    assert bs_dict['beliefstate'] == bst.bs
    return bs_dict


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
    previous_bs = bst.bs
    bs_dict = execute_update_bst(bst, None)
    assert bs_dict['beliefstate'] == previous_bs


def test_reset_requests_on_update(bst):
    """
    Tests whether the request dict resets when updating the BST.

    Args:
        bst: BST Object (given in conftest.py)
    """
    bst.bs['requests']['foo'] = 0.5
    user_acts = [UserAct(act_type=UserActionType.Hello)]
    bs_dict = execute_update_bst(bst, user_acts)
    assert bs_dict['beliefstate']['requests'] == {}


def test_set_user_action_types_on_update(bst):
    """
    Tests whether the user action types are set correctly when updating the BST.

    Args:
        bst: BST Object (given in conftest.py)
    """
    user_action_types = [UserActionType.Inform, UserActionType.Request, UserActionType.Hello,
                         UserActionType.Thanks]
    user_acts = [UserAct(act_type=act_type) for act_type in user_action_types]
    bs_dict = execute_update_bst(bst, user_acts)
    assert bs_dict['beliefstate']['user_acts'] == set(user_action_types)


def test_reset_informs_about_primary_key_on_update_with_user_inform(bst):
    """
    Tests whether the informs about the primary key reset when updating the BST with a user inform.

    Args:
        bst: BST Object (given in conftest.py)
    """
    bst.bs['informs'][bst.domain.get_primary_key()] = {'foo': 0.5}
    user_acts = [UserAct(act_type=UserActionType.Inform)]
    bs_dict = execute_update_bst(bst, user_acts)
    assert bst.domain.get_primary_key() not in bs_dict['beliefstate']['informs']


def test_reset_informs_and_request_on_update_with_user_select_domain(bst):
    """
    Tests whether the informs and requests reset when updating the BST with a new domain
    selection by the user.

    Args:
        bst: BST Object (given in conftest.py)
    """
    bst.bs['informs']['foo'] = {'bar': 0.5}
    bst.bs['requests']['foo'] = 0.5
    user_acts = [UserAct(act_type=UserActionType.SelectDomain)]
    bs_dict = execute_update_bst(bst, user_acts)
    assert bs_dict['beliefstate']['informs'] == {}
    assert bs_dict['beliefstate']['requests'] == {}


def test_update_bst_with_user_request(bst):
    """
    Tests whether the requests are set correctly when updating the BST with a user request.

    Args:
        bst: BST Object (given in conftest.py)
    """
    slot = 'foo'
    score = 0.5
    user_acts = [UserAct(act_type=UserActionType.Request, slot=slot, score=score)]
    bs_dict = execute_update_bst(bst, user_acts)
    assert slot in bs_dict['beliefstate']['requests']
    assert bs_dict['beliefstate']['requests'][slot] == score


def test_update_bst_with_user_inform(bst):
    """
    Tests whether the informs are set correctly when updating the BST with a user inform.

    Args:
        bst: BST Object (given in conftest.py)
    """
    slot = 'foo'
    value = 'bar'
    score = 0.5
    user_acts = [UserAct(act_type=UserActionType.Inform, slot=slot, value=value, score=score)]
    bs_dict = execute_update_bst(bst, user_acts)
    assert slot in bs_dict['beliefstate']['informs']
    assert value in bs_dict['beliefstate']['informs'][slot]
    assert bs_dict['beliefstate']['informs'][slot][value] == score


def test_update_bst_with_user_inform_resets_inform(bst):
    """
    Tests whether updating the BST with a user inform resets the previous inform.

    Args:
        bst: BST Object (given in conftest.py)
    """
    slot = 'foo'
    previous_slot_inform = {'bar': 0.5}
    bst.bs['informs'][slot] = previous_slot_inform
    user_acts = [UserAct(act_type=UserActionType.Inform, slot=slot, value='baz', score=0.3)]
    bs_dict = execute_update_bst(bst, user_acts)
    assert bs_dict['beliefstate']['informs'][slot] != previous_slot_inform


def test_update_bst_with_user_negative_inform(bst):
    """
    Tests whether updating the BST with a negative inform by the user deletes the corresponding
    inform value.

    Args:
        bst: BST Object (given in conftest.py)
    """
    slot = 'foo'
    value = 'bar'
    bst.bs['informs'][slot] = {value: 0.5}
    user_acts = [UserAct(act_type=UserActionType.NegativeInform, slot=slot, value=value)]
    bs_dict = execute_update_bst(bst, user_acts)
    assert value not in bs_dict['beliefstate']['informs'][slot]


def test_update_bst_with_user_request_alternatives(bst, primkey_constraint):
    """
    Tests whether the primary key is removed from the informs when updating the BST with a user
    request for alternatives.

    Args:
        bst: BST Object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    bst.bs['informs'][primkey_constraint['slot']] = {primkey_constraint['value']: 0.5}
    user_acts = [UserAct(act_type=UserActionType.RequestAlternatives)]
    bs_dict = execute_update_bst(bst, user_acts)
    assert bst.domain.get_primary_key() not in bs_dict['beliefstate']['informs']


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
    bs_dict = execute_update_bst(bst, user_acts)
    assert 'num_matches' in bs_dict['beliefstate']
    assert type(bs_dict['beliefstate']['num_matches']) == int
    assert bs_dict['beliefstate']['num_matches'] > -1
    assert 'discriminable' in bs_dict['beliefstate']
    assert type(bs_dict['beliefstate']['discriminable']) == bool


