import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from utils import SysAct, UserAct, SysActionType, UserActionType
from services.simulator.goal import Constraint


# Test dialog_start()

def test_reset_goal_on_dialog_start(simulator):
    """
    Tests whether starting the dialog reinits the goal.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    constraints = ['foo']
    requests = {'foo': 'bar'}
    simulator.goal.constraints = constraints.copy()
    simulator.goal.requests = requests.copy()
    simulator.dialog_start()
    assert simulator.goal.constraints != constraints
    assert simulator.goal.requests != requests


def test_reset_agenda_on_dialog_start(simulator):
    """
    Test whether starting the dialog resets the agenda.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    stack = [UserAct(), UserAct()]
    simulator.agenda.stack = stack.copy()
    simulator.dialog_start()
    assert simulator.agenda.stack != stack


def test_reset_fixed_patience_on_dialog_start(simulator):
    """
    Tests whether starting the dialog resets the patience to a fixed value if it is speficied
    like this in the config file.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    patience = 7
    simulator.parameters['usermodel']['patience'] = [patience]
    simulator.patience = 4
    simulator.dialog_patience = 6
    simulator.dialog_start()
    assert simulator.dialog_patience == patience
    assert simulator.patience == patience


def test_reset_random_patience_on_dialog_start(simulator):
    """
    Tests whether starting the dialog resets the patience to a random value within a certain
    range if it is speficied like this in the config file.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    lower_bound = 3
    upper_bound = 7
    simulator.parameters['usermodel']['patience'] = [lower_bound, upper_bound]
    simulator.patience = 4
    simulator.dialog_patience = 6
    simulator.dialog_start()
    assert simulator.dialog_patience <= upper_bound
    assert simulator.dialog_patience >= lower_bound
    assert simulator.patience == simulator.dialog_patience


def test_reset_actions_on_dialog_start(simulator):
    """
    Tests whether starting the dialog resets the list of last user actions and the last system
    action. Both should be reset to None.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.last_user_actions = [UserAct()]
    simulator.last_system_action = SysAct()
    simulator.dialog_start()
    assert simulator.last_user_actions is None
    assert simulator.last_system_action is None


def test_reset_excluded_venues_on_dialog_start(simulator, primkey_constraint):
    """
    Tests whether starting the dialog resets the list of excluded venues to an empty list.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    simulator.excluded_venues = [primkey_constraint['value']]
    simulator.dialog_start()
    assert simulator.excluded_venues == []


def test_reset_turn_on_dialog_start(simulator):
    """
    Tests whether starting the dialog resets the turn number to zero.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.turn = 10
    simulator.dialog_start()
    assert simulator.turn == 0


# Test user_turn()

def test_user_turn_on_system_bye(simulator):
    """
    Tests whether the simulator returns the goal when the system action is Bye.


    Args:
        simulator: Simulator object (given in conftest.py)
    """
    res = simulator.user_turn(sys_act=SysAct(act_type=SysActionType.Bye))
    assert type(res) == dict
    assert len(res) == 1
    assert 'sim_goal' in res
    assert res['sim_goal'] == simulator.goal


def test_user_turn_with_sys_act(simulator, constraintA):
    """
    Tests the returned user turn in case of a given system action. In this case, the system
    action is Request which should trigger a user inform action. In contrast to a user turn
    without given system action, actions from the agenda that were added in previous turns should only be returned if several user actions are selected for one turn.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    stack = [UserAct(act_type=UserActionType.Confirm)]
    simulator.agenda.stack = stack.copy()
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    simulator.num_actions_next_turn = 1
    sys_act = SysAct(act_type=SysActionType.Request, slot_values={constraintA['slot']: []})
    res = simulator.user_turn(sys_act=sys_act)
    assert type(res) == dict
    assert len(res) == 1
    assert 'user_acts' in res
    assert len(res['user_acts']) == 1
    assert res['user_acts'][0].type == UserActionType.Inform


def test_user_turn_without_sys_act(simulator):
    """
    Tests the returned user turn in case of a no given system action. The user action should now
    be an action that was added to the agenda in a previous turn.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    # respond
    stack = [UserAct(act_type=UserActionType.Confirm)]
    simulator.agenda.stack = stack.copy()
    simulator.num_actions_next_turn = 1
    res = simulator.user_turn(sys_act=None)
    assert type(res) == dict
    assert len(res) == 1
    assert 'user_acts' in res
    assert res['user_acts'] == stack


# Test receive()

def test_receive_reduce_patience(simulator):
    """
    Tests whether the patience is reduced by one if the system performed the same action as in
    the previous turn.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    sys_act = SysAct(act_type=SysActionType.InformByName, slot_values={'foo': ['bar']})
    patience = 7
    simulator.patience = patience
    simulator.last_system_action = sys_act
    simulator.receive(sys_act)
    assert simulator.patience == patience - 1


def test_receive_resets_patience(simulator):
    """
    Tests whether the patience is reset to the default value if the system action is different
    from the one of the previous turn.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.dialog_patience = 7
    simulator.patience = 3
    simulator.parameters['usermodel']['resetPatience'] = 1
    simulator.last_system_action = SysAct(act_type=SysActionType.InformByName)
    simulator.receive(SysAct(act_type=SysActionType.Request))
    assert simulator.patience == simulator.dialog_patience


def test_receive_without_patience(simulator):
    """
    Tests whether the simulator ends the dialog if its patience reaches zero.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.last_system_action = None
    simulator.patience = 0
    simulator.receive(SysAct(act_type=SysActionType.InformByName))
    assert simulator.agenda.stack[-1].type == UserActionType.Bye


def test_receive_on_invalid_system_act(simulator):
    """
    Tests whether the simulator aborts the processing of the system action if the action type is
    unknown. In this case, the agenda should not change within the receive method.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        caplog: pytest fixture for testing the logging messages
    """
    simulator.last_system_action = None
    simulator.patience = 10
    stack = [UserAct(act_type=UserActionType.Confirm)]
    simulator.agenda.stack = stack.copy()
    simulator.receive(SysAct('foo'))
    assert simulator.agenda.stack == stack


def test_receive_with_ignored_requests(simulator, constraintA):
    """
    Tests whether requests from the last turn of the simulator remained unanswered by the
    system. In this case, those requests should be repeated.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    request = UserAct(act_type=UserActionType.Request, slot=constraintA['slot'])
    simulator.last_user_actions = [request]
    simulator.receive(SysAct(act_type=SysActionType.InformByName))
    assert simulator.agenda.stack[-1] == request


def test_receive_with_ignored_alternative_requests(simulator, constraintA):
    """
    Tests whether requests for alternatives from the last turn of the simulator remained unanswered
    by the system. In this case, those requests should be repeated.


    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    alt_request = UserAct(act_type=UserActionType.RequestAlternatives, slot=constraintA['slot'])
    simulator.last_user_actions = [UserAct(act_type=UserActionType.Request, slot=constraintA[
        'slot']), alt_request]
    simulator.receive(SysAct(act_type=SysActionType.InformByName))
    assert simulator.agenda.stack[-1] == alt_request
    assert simulator.num_actions_next_turn == len([alt_request])
    assert all(item.type != UserActionType.Request for item in simulator.agenda.stack)


def test_receive_on_fulfilled_goal(simulator, primkey_constraint):
    """
    Tests whether the simulator finishes the dialog if the system could fulfil the goal in its
    last turn and there are no further items on the simulator's agenda.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    simulator.goal.requests = {primkey_constraint['slot']: primkey_constraint['value']}
    simulator.agenda.stack = []
    simulator.receive(SysAct(act_type=SysActionType.Welcome))
    assert simulator.goal.is_fulfilled()
    assert simulator.agenda.stack[-1].type in (UserActionType.Bye, UserActionType.Thanks)


# Test respond()

def test_respond_with_empty_agenda(simulator):
    """
    Tests whether an empty agenda in the respond method raises an Assertion Error.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.agenda.stack = []
    with pytest.raises(AssertionError):
        simulator.respond()


def test_respond_with_several_actions(simulator):
    """
    Tests the number and order of responded actions in case of several actions shpuld be returned
    (controlled by num_actions_next_turn)

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    stack = [UserAct(act_type=UserActionType.Inform),
             UserAct(act_type=UserActionType.RequestAlternatives),
             UserAct(act_type=UserActionType.Confirm),
             UserAct(act_type=UserActionType.Request)]
    simulator.agenda.stack = stack.copy()
    num_actions = 3
    simulator.num_actions_next_turn = num_actions
    res = simulator.respond()
    assert len(res) == num_actions
    assert res == stack[-1:(len(stack)-(num_actions+1)):-1]
    assert simulator.num_actions_next_turn == -1


def test_respond_with_bye(simulator):
    """
    Tests the respond in case the selected user action is a Bye.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    stack = [UserAct(act_type=UserActionType.Bye)]
    simulator.agenda.stack = stack.copy()
    simulator.num_actions_next_turn = 0
    res = simulator.respond()
    assert res == stack


def test_respond_with_random_number_of_actions(simulator):
    """
    Tests the number of responded actions in case a random number is selected.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.agenda.stack = [UserAct(act_type=UserActionType.Inform),
                              UserAct(act_type=UserActionType.RequestAlternatives),
                              UserAct(act_type=UserActionType.Confirm),
                              UserAct(act_type=UserActionType.Request)]
    simulator.num_actions_next_turn = 0
    res = simulator.respond()
    assert len(res) > 0
    assert len(res) < 4


def test_respond_with_missing_inform(simulator):
    """
    Tests whether responded inform actions are removed from the missing informs list of the goal.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    action = UserAct(act_type=UserActionType.Inform)
    simulator.agenda.stack = [action]
    simulator.goal.missing_informs = [action]
    simulator.num_actions_next_turn = 1
    res = simulator.respond()
    assert action not in simulator.goal.missing_informs
    assert action in res


# Test _receive_*() methods

def test_receive_informbyname_fulfills_goal(simulator, primkey_constraint, constraintA):
    """
    Tests the case if an inform by name action by the system fulfills the goal.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    slot_values = {
        primkey_constraint['slot']: [primkey_constraint['value']],
        constraintA['slot']: [constraintA['value']]
    }
    sys_act = SysAct(act_type=SysActionType.InformByName, slot_values=slot_values)
    simulator.goal.requests = {
        primkey_constraint['slot']: primkey_constraint['value'],
        constraintA['slot']: None
    }
    simulator.goal.constraints = []
    simulator.agenda.stack = []
    simulator._receive_informbyname(sys_act)
    assert simulator.goal.is_fulfilled()
    assert simulator.agenda.stack[-1].type in (UserActionType.Bye, UserActionType.Thanks)


def test_receive_informbyname_fulfills_request_but_not_goal(simulator,
                                                                      primkey_constraint,
                                                            constraintA, constraintB):
    """
    Tests the case if an inform by name action by the system fulfills a request but not the
    complete goal.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    slot_values = {
        primkey_constraint['slot']: [primkey_constraint['value']],
        constraintA['slot']: [constraintA['value']]
    }
    sys_act = SysAct(act_type=SysActionType.InformByName, slot_values=slot_values)
    simulator.goal.requests = {
        primkey_constraint['slot']: primkey_constraint['value'],
        constraintA['slot']: None,
        constraintB['slot']: None
    }
    simulator.goal.constraints = []
    simulator._receive_informbyname(sys_act)
    assert not simulator.goal.is_fulfilled()
    assert constraintA['value'] in simulator.goal.requests[constraintA['slot']]


def test_receive_informbyalternatives_without_fulfilled_primkey(simulator, primkey_constraint):
    """
    Tests the case if an inform by alternatives action by the system without a previously
    fulfilled primary key request in the goal. This simply processes the action as if it were an
    inform by name action.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    simulator.last_user_actions = [UserAct(act_type=UserActionType.Inform),
                                   UserAct(act_type=UserActionType.Request)]
    simulator.agenda.stack = []
    simulator.excluded_venues = [primkey_constraint['value']]
    simulator.goal.requests = {primkey_constraint['slot']: None}
    sys_act = SysAct(act_type=SysActionType.InformByAlternatives)
    simulator._receive_informbyalternatives(sys_act)
    assert simulator.agenda.stack[-len(simulator.last_user_actions):] != \
           simulator.last_user_actions[::-1]


def test_receive_informbyalternatives_with_fulfilled_primkey(simulator, primkey_constraint):
    """
    Tests the case if an inform by alternatives action by the system with a previously
    fulfilled primary key request in the goal. This simply repeats the last user actions.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    simulator.last_user_actions = [UserAct(act_type=UserActionType.Inform),
                                   UserAct(act_type=UserActionType.Request)]
    simulator.goal.requests = {primkey_constraint['slot']: primkey_constraint['value']}
    sys_act = SysAct(act_type=SysActionType.InformByAlternatives)
    simulator._receive_informbyalternatives(sys_act)
    assert simulator.agenda.stack[-len(simulator.last_user_actions):] == \
           simulator.last_user_actions[::-1]


def test_receive_request(simulator, constraintA):
    """
    Tests the case of a request action by the system.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    sys_act = SysAct(act_type=SysActionType.Request, slot_values={constraintA['slot']: []})
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    simulator._receive_request(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.Inform
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value == constraintA['value']


def test_receive_confirm_with_inconsistent_constraint_triggers_inform(simulator,  constraintA,
                                                                      constraintA_alt):
    """
    Tests the case of a confirm action by the system which confirms something that is
    inconsistent to the goal. With some probability, the simulator will inform about the actual
    value in the goal.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA_alt['slot'], constraintA_alt['value'])]
    simulator.parameters['usermodel']['InformOnConfirm'] = 2
    sys_act = SysAct(act_type=SysActionType.Confirm, slot_values={constraintA['slot']: [constraintA['value']]})
    simulator._receive_confirm(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.Inform
    assert simulator.agenda.stack[-1].slot == constraintA_alt['slot']
    assert simulator.agenda.stack[-1].value == constraintA_alt['value']


def test_receive_confirm_with_inconsistent_constraint_triggers_negative_inform(simulator,
                                                                               constraintA,
                                                                               constraintA_alt):
    """
    Tests the case of a confirm action by the system with confirms something that is inconsistent to
    the goal. The simulator will perform a negative inform about that constraint.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA_alt['slot'], constraintA_alt['value'])]
    simulator.parameters['usermodel']['InformOnConfirm'] = -1
    sys_act = SysAct(act_type=SysActionType.Confirm,
                     slot_values={constraintA['slot']: [constraintA['value']]})
    simulator._receive_confirm(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.NegativeInform
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value == constraintA['value']


def test_receive_confirm_without_inconsistent_constraints(simulator, constraintA):
    """
    Tests the case of a confirm action by the system which informs about something that is
    consistent to the goal.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    sys_act = SysAct(act_type=SysActionType.Confirm,
                     slot_values={constraintA['slot']: [constraintA['value']]})
    simulator._receive_confirm(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.Inform
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value == constraintA['value']


def test_receive_select_with_inconsistent_constraint_triggers_inform(simulator, constraintA,
                                                                     constraintA_alt):
    """
    Tests the case of a select action by the system which selects something that is inconsistent
    to the goal. With some probability, the simulator will then inform about the actual value in
    the goal, before performing negative inform actions on all values that were selected by the
    system.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA_alt['slot'], constraintA_alt['value'])]
    simulator.parameters['usermodel']['InformOnSelect'] = 2
    sys_act = SysAct(act_type=SysActionType.Select,
                     slot_values={constraintA['slot']: [constraintA['value']]})
    simulator._receive_select(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.NegativeInform
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value == constraintA['value']
    assert simulator.agenda.stack[-2].type == UserActionType.Inform
    assert simulator.agenda.stack[-2].slot == constraintA_alt['slot']
    assert simulator.agenda.stack[-2].value == constraintA_alt['value']


def test_receive_select_with_inconsistent_constraint_triggers_negative_inform(simulator,
                                                                              constraintA,
                                                                              constraintA_alt):
    """
    Tests the case of a select action by the system which selects something that is inconsistent
    to the goal. The simulator perform negative inform actions on all values that were selected
    by the system.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA_alt['slot'], constraintA_alt['value'])]
    simulator.parameters['usermodel']['InformOnSelect'] = -1
    sys_act = SysAct(act_type=SysActionType.Select,
                     slot_values={constraintA['slot']: [constraintA['value']]})
    simulator._receive_select(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.NegativeInform
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value == constraintA['value']


def test_receive_select_without_inconsistent_constraint(simulator, constraintA):
    """
    Tests the case of a select action by the system which selects something that is consistent to the goal. It will then process the action is if it was a request.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    sys_act = SysAct(act_type=SysActionType.Select,
                     slot_values={constraintA['slot']: [constraintA['value']]})
    simulator._receive_select(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.Inform
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value == constraintA['value']


def test_receive_requestmore_with_fulfilled_goal(simulator, constraintA):
    """
    Tests the case of a request more action by the system that fulfills the goal. This finishes
    the dialog.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
    """
    simulator.goal.requests = {constraintA['slot']: constraintA['value']}
    sys_act = SysAct(act_type=SysActionType.RequestMore)
    simulator._receive_requestmore(sys_act)
    assert simulator.goal.is_fulfilled()
    assert simulator.agenda.stack[-1].type in (UserActionType.Bye, UserActionType.Thanks)


def test_receive_requestmore_with_partly_fulfilled_goal(simulator, constraintA, primkey_constraint):
    """
    Tests the case of a request more action by the system that does not fulfil the complete goal
    but in which case the primary key request of the goal is already filled and all informs in
    the agenda of the simulator have been performed. The simulator will the perform requests
    about the still open slots.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    simulator.goal.requests = {
        primkey_constraint['slot']: primkey_constraint['value'],
        constraintA['slot']: None
    }
    simulator.agenda.stack = []
    sys_act = SysAct(act_type=SysActionType.RequestMore)
    simulator._receive_requestmore(sys_act)
    assert not simulator.goal.is_fulfilled()
    assert simulator.agenda.stack[-1].type == UserActionType.Request
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value == None


def test_receive_requestmore_with_open_goal(simulator, primkey_constraint, constraintA):
    """
    Tests the case of a request more action by the system which does not fulfil the goal and in
    which the goal is not even close to be fulfilled. The simulator will then repeat its last
    actions.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.last_user_actions = [UserAct(act_type=UserActionType.Inform), UserAct(
        act_type=UserActionType.Request)]
    simulator.goal.requests = {
        primkey_constraint['slot']: None,
        constraintA['slot']: None
    }
    sys_act = SysAct(act_type=SysActionType.RequestMore)
    simulator._receive_requestmore(sys_act)
    assert simulator.agenda.stack[-len(simulator.last_user_actions):] == \
           simulator.last_user_actions[::-1]


def test_receive_bad(simulator):
    """
    Test the case of a bad action by the system. The simulator will then repeat its last actions.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.last_user_actions = [UserAct(act_type=UserActionType.Inform), UserAct(
        act_type=UserActionType.Request)]
    sys_act = SysAct(act_type=SysActionType.Bad)
    simulator._receive_bad(sys_act)
    assert simulator.agenda.stack[-len(simulator.last_user_actions):] == \
           simulator.last_user_actions[::-1]


def test_receive_confirmrequest_with_missing_values(simulator, constraintA):
    """
    Tests the case of a confirm request action by the system in which at least one value is
    empty. The simulator will handle this action as a request.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    sys_act = SysAct(act_type=SysActionType.ConfirmRequest, slot_values={constraintA['slot']: None})
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    simulator._receive_confirmrequest(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.Inform
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value == constraintA['value']


def test_receive_confirmrequest_with_filled_values(simulator, constraintA, constraintA_alt):
    """
    Tests the case of a confirm request action by the system in which no value is empty. The
    simulator will handle this action as a confirm action.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
    """
    sys_act = SysAct(act_type=SysActionType.ConfirmRequest, slot_values={constraintA['slot']: constraintA[
                                                                                        'value']})
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    simulator.goal.requests = {constraintA_alt['slot']: constraintA_alt['value']}
    simulator._receive_confirmrequest(sys_act)
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type in (UserActionType.Inform, UserActionType.NegativeInform)
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value in (constraintA['value'], constraintA_alt['value'])


# Test other helper methods

def test_finish_dialog_grateful(simulator):
    """
    Tests the case the simulator finishes the dialog grateful by saying thanks and bye.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.agenda.stack = [UserAct()]
    simulator.parameters['usermodel']['Thank'] = 2
    simulator._finish_dialog(ungrateful=False)
    assert len(simulator.agenda.stack) == 2
    assert simulator.agenda.stack[0].type == UserActionType.Thanks
    assert simulator.agenda.stack[1].type == UserActionType.Bye


@pytest.mark.parametrize('on_random', [True, False])
def test_finish_dialog_ungrateful(simulator, on_random):
    """
    Tests the case the simulator finishes the dialog ungrateful by saying simply bye without
    thanking. This can occur in two situation which are tested by setting the parameter
    on_random. It is either selected by some probability randomly, or fixed in the function call.

    Args:
        simulator: Simulator object (given in conftest.py)
        on_random (bool): bool to decide whether to finish the dialog ungrateful on random or
        planned
    """
    simulator.agenda.stack = [UserAct()]
    if on_random:
        simulator.parameters['usermodel']['Thank'] = -1
        simulator._finish_dialog(ungrateful=False)
    else:
        simulator._finish_dialog(ungrateful=True)
    assert len(simulator.agenda.stack) == 1
    assert simulator.agenda.stack[0].type == UserActionType.Bye


def test_repeat_last_actions(simulator):
    """
    Tests the case the simulator repeats its last actions.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    simulator.last_user_actions = [UserAct(act_type=UserActionType.Inform), UserAct(
        act_type=UserActionType.Request)]
    simulator._repeat_last_actions()
    assert simulator.num_actions_next_turn == len(simulator.last_user_actions)
    assert simulator.agenda.stack[-len(simulator.last_user_actions):] == \
           simulator.last_user_actions[::-1]


def test_alter_constraints_without_explicit_constraints(simulator, constraintA):
    """
    Tests altering constraints in the goal in which a random valid value is selected for one or
    several slots in the goal's constraints. If no constraints are given explicitly for this
    process, random constraints from the goal are selected.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    res = simulator._alter_constraints([], 1)
    assert len(res) == len(simulator.goal.constraints)
    assert res[0].slot == constraintA['slot']
    assert res[0].value != constraintA['value']
    assert simulator.goal.constraints != [Constraint(constraintA['slot'], constraintA['value'])]


def test_alter_constraints_with_inconsistent_constraints(simulator, constraintA, constraintA_alt):
    """
    Tests altering constraints in the goal in which a random valid value is selected for one or
    several slots in the goal's constraints. If all given constraints are not part of the goal,
    no constraints will be changed.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA_alt['slot'], constraintA_alt['value'])]
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    res = simulator._alter_constraints(constraints, 1)
    assert res == []


def test_alter_constraints_without_possible_values(simulator, constraintA):
    """
    Tests altering constraints in the goal in which a random valid value is selected for one or
    several slots in the goal's constraints. If there are no other values possible for a slot,
    the value will be changed to "dontcare".

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    simulator.goal.excluded_inf_slot_values[constraintA['slot']] = \
        set(simulator.domain.get_possible_values(constraintA['slot']))
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    res = simulator._alter_constraints(constraints.copy(), 1)
    assert len(res) == len(constraints)
    assert res[0].slot == constraintA['slot']
    assert res[0].value == 'dontcare'
    assert Constraint(constraintA['slot'], 'dontcare') in simulator.goal.constraints


def test_alter_constraints_into_dontcare(simulator, constraintA):
    """
    Tests altering constraints in the goal in which a random valid value is selected for one or
    several slots in the goal's constraints. Even if there are other values possible for a slot,
    with a certain probability the value will be changed into "dontcare".

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    simulator.parameters['usermodel']['DontcareIfNoVenue'] = 2
    res = simulator._alter_constraints(constraints.copy(), 1)
    assert len(res) == len(constraints)
    assert res[0].slot == constraintA['slot']
    assert res[0].value == 'dontcare'
    assert Constraint(constraintA['slot'], 'dontcare') in simulator.goal.constraints


def test_alter_constraints_on_random(simulator, constraintA):
    """
    Tests altering constraints in the goal in which a random valid value is selected for one or
    several slots in the goal's constraints. This selects a random value of all possible values
    for a slot.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    res = simulator._alter_constraints(constraints.copy(), 1)
    assert len(res) == len(constraints)
    assert res[0].slot == constraintA['slot']
    assert res[0].value != constraintA['value']
    assert simulator.goal.constraints != [Constraint(constraintA['slot'], constraintA['value'])]


def test_check_informs_with_inconsistent_constraint(simulator, constraintA, constraintA_alt):
    """
    Tests the check for consistency of an inform with the goal in case the inform is not
    consistent with the goal. The simulator will then inform about the actual value.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA_alt['slot'], constraintA_alt['value'])]
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    is_consistent_with_goal = simulator._check_informs(constraints)
    assert is_consistent_with_goal is False
    assert len(simulator.agenda) > 0
    assert simulator.agenda.stack[-1].type == UserActionType.Inform
    assert simulator.agenda.stack[-1].slot == constraintA_alt['slot']
    assert simulator.agenda.stack[-1].value == constraintA_alt['value']


def test_check_informs_without_inconsistent_constraint(simulator, constraintA):
    """
    Tests the check for consistency of an inform with the goal in case the inform is indeed
    consistent with the goal. The simulator will not inform about it again.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    simulator.agenda.stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                                      value=constraintA['value'])]
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    is_consistent_with_goal = simulator._check_informs(constraints)
    assert is_consistent_with_goal is True
    assert not any(item.type == UserActionType.Inform and item.slot == constraintA['slot']
                   and item.value == constraintA['value'] for item in simulator.agenda.stack)


def test_check_offer_with_inconsistent_informs(simulator, constraintA, constraintA_alt,
                                               primkey_constraint):
    """
    Tests the check for validity of an offer. An offer is a value for the primary key slot of
    the domain the system has informed about. In case the offer was given with informs that are
    inconsistent to the goal, the offer is declined.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    offers = []
    simulator.goal.constraints = [Constraint(constraintA_alt['slot'], constraintA_alt['value'])]
    simulator.goal.requests[primkey_constraint['slot']] = 'foo'
    is_valid_offer = simulator._check_offer(offers, constraints)
    assert is_valid_offer is False
    assert simulator.goal.requests[primkey_constraint['slot']] is None


def test_check_offer_with_too_early_offer(simulator, primkey_constraint,
                                                         constraintA, constraintB):
    """
    Tests the check for validity of an offer. An offer is a value for the primary key slot of
    the domain the system has informed about. In case this offer is given too early and the
    simulator has still not informed about everything, the offer is declined.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    constraints = []
    offers = [primkey_constraint['value']]
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value']),
                                  Constraint(constraintB['slot'], constraintB['value'])]
    simulator.agenda.stack = [UserAct(act_type=UserActionType.Inform), UserAct(
        act_type=UserActionType.Inform)]
    simulator.last_user_actions = [UserAct(act_type=UserActionType.Inform), UserAct(
        act_type=UserActionType.Request)]
    is_valid_offer = simulator._check_offer(offers, constraints)
    assert is_valid_offer is False
    assert simulator.num_actions_next_turn == len(simulator.last_user_actions)
    assert simulator.agenda.stack[-len(simulator.last_user_actions):] == \
           simulator.last_user_actions[::-1]


@pytest.mark.parametrize('offers_in_excluded_venues', [True, False])
def test_check_offer_with_request_for_alternatives(simulator,
                                                                       primkey_constraint,
                                                         constraintA, offers_in_excluded_venues):
    """
    Tests the check for validity of an offer. An offer is a value for the primary key slot of
    the domain the system has informed about. If the primary key slot in the goal is still open,
    there are two situations in which the simulator will request an alternative offer. Those
    situations are controlled by the parameter offers_in_excluded_venues in this test. It is
    either selected by chance although the offer is valid or it is not a valid offer because it
    was previously excluded.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        offers_in_excluded_venues (bool): whether to set offers in excluded venues or not
    """
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    offers = [primkey_constraint['value']]
    if offers_in_excluded_venues:
        simulator.excluded_venues = offers
    simulator.parameters['usermodel']['ReqAlt'] = 2
    simulator.goal.constraints = []
    simulator.goal.requests = {
        primkey_constraint['slot']: None,
        constraintA['slot']: constraintA['value']
    }
    is_valid_offer = simulator._check_offer(offers, constraints)
    assert is_valid_offer is False
    assert all(value is None for value in simulator.goal.requests.values())
    assert simulator.agenda.stack[-1].type == UserActionType.RequestAlternatives


def test_check_offer_triggering_goal_change(simulator, primkey_constraint,
                                                         constraintA):
    """
    Tests the check for validity of an offer. An offer is a value for the primary key slot of
    the domain the system has informed about. If the offer is 'none', the goal is changed.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    offers = ['none']
    simulator.goal.requests = {
        primkey_constraint['slot']: None,
        constraintA['slot']: constraintA['value']
    }
    simulator.goal.constraints = [Constraint(constraintA['slot'], constraintA['value'])]

    is_valid_offer = simulator._check_offer(offers, constraints)
    assert is_valid_offer is False
    assert all(value is None for value in simulator.goal.requests.values())
    assert simulator.goal.missing_informs[0].type == UserActionType.Inform
    assert simulator.goal.missing_informs[0].slot == constraintA['slot']
    assert simulator.goal.missing_informs[0].value != constraintA['value']
    assert simulator.agenda.stack[-1].type == UserActionType.Inform
    assert simulator.agenda.stack[-1].slot == constraintA['slot']
    assert simulator.agenda.stack[-1].value != constraintA['value']


def test_check_offer_without_matching_offer(simulator, primkey_constraint,
                                                         constraintA):
    """
    Tests the check for validity of an offer. An offer is a value for the primary key slot of
    the domain the system has informed about. If the primary key slot is already filled in the
    goal but the offer does not match it, it is declined.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    offers = ['foo']
    simulator.goal.requests = {
        primkey_constraint['slot']: 'bar',
        constraintA['slot']: constraintA['value']
    }
    simulator.goal.constraints = []
    is_valid_offer = simulator._check_offer(offers, constraints)
    assert is_valid_offer is False


def test_check_offer_for_valid_filled_offer(simulator, constraintA, primkey_constraint):
    """
    Tests the check for validity of an offer. An offer is a value for the primary key slot of
    the domain the system has informed about. If the primary key slot is already filled in the
    goal and the offer matches it, the offer is accepted.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    offers = [primkey_constraint['value']]
    simulator.goal.requests = {
        primkey_constraint['slot']: primkey_constraint['value'],
        constraintA['slot']: constraintA['value']
    }
    simulator.goal.constraints = []
    is_valid_offer = simulator._check_offer(offers, constraints)
    assert is_valid_offer is True


def test_check_offer_for_valid_unfilled_offer(simulator, constraintA, primkey_constraint,
                                              constraintB):
    """
    Tests the check for validity of an offer. An offer is a value for the primary key slot of
    the domain the system has informed about. If the primary key slot is not filled in the goal
    but the offer has not been excluded before, it is accepted as valid offer.


    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    constraints = [Constraint(constraintA['slot'], constraintA['value'])]
    offers = [primkey_constraint['value']]
    simulator.goal.requests = {
        primkey_constraint['slot']: None,
        constraintA['slot']: constraintA['value']
    }
    simulator.goal.constraints = []
    simulator.goal.missing_informs = [UserAct(act_type=UserActionType.Inform, slot=constraintB[
        'slot'], value=constraintB['value'])]
    simulator.parameters['usermodel']['ReqAlt'] = -1
    is_valid_offer = simulator._check_offer(offers, constraints)
    assert is_valid_offer is True
    assert simulator.goal.requests[primkey_constraint['slot']] == offers[0]
    assert simulator.agenda.stack[-1].type == UserActionType.Request
    assert simulator.agenda.stack[-1].slot == constraintB['slot']
    assert simulator.agenda.stack[-1].value is None


def test_request_alt(simulator, primkey_constraint, constraintA):
    """
    Tests the case the simulator requests an alternative offer. This resets the goal.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    simulator.goal.requests = {
        constraintA['slot']: constraintA['value'],
        primkey_constraint['slot']: None
    }
    simulator._request_alt(offer=None)
    assert all(value is None for value in simulator.goal.requests.values())
    assert simulator.agenda.stack[-1].type == UserActionType.RequestAlternatives


def test_request_alt_with_offer(simulator, primkey_constraint):
    """
    Tests the case the simulator requests an alternative offer for a specific, previously given
    offer. The previous offer will be excluded.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    offer = primkey_constraint['value']
    simulator.goal.requests = {
        primkey_constraint['slot']: None
    }
    simulator._request_alt(offer=offer)
    assert offer in simulator.excluded_venues
    assert all(value is None for value in simulator.goal.requests.values())
    assert simulator.agenda.stack[-1].type == UserActionType.RequestAlternatives


def test_request_alt_with_filled_primkey_request(simulator, primkey_constraint):
    """
    Tests the case the simulator requests an alternative offer if the primary key slot in the
    goal is already filled. The slot will then be emptied in the goal and the previous value
    excluded.

    Args:
        simulator: Simulator object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    primkey_value = primkey_constraint['value']
    simulator.goal.requests = {
        primkey_constraint['slot']: primkey_value
    }
    simulator._request_alt(offer=None)
    assert primkey_value in simulator.excluded_venues
    assert simulator.goal.requests[primkey_constraint['slot']] is None
    assert all(value is None for value in simulator.goal.requests.values())
    assert simulator.agenda.stack[-1].type == UserActionType.RequestAlternatives


def test_check_system_ignored_request_without_user_actions(simulator):
    """
    Tests the check for ignored requests by the system in case no user actions are given. In that case, no requests were ignored.

    Args:
        simulator: Simulator object (given in conftest.py)
    """
    user_actions = []
    sys_act = SysAct()
    res = simulator._check_system_ignored_request(user_actions, sys_act)
    assert type(res) == tuple
    assert len(res) == 2
    assert res == ([], [])


def test_check_system_ignored_request_for_informbyname(simulator, constraintA, constraintB):
    """
    Tests the check for ignored requests by the system in an inform by name action.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    user_actions = [UserAct(act_type=UserActionType.Request, slot=constraintA['slot'])]
    sys_act = SysAct(act_type=SysActionType.InformByName, slot_values={constraintB['slot']: []})
    res = simulator._check_system_ignored_request(user_actions, sys_act)
    assert type(res) == tuple
    assert len(res) == 2
    assert res[0] == user_actions


def test_check_system_ignored_request_for_informbyalternatives(simulator, constraintA,
                                                               primkey_constraint):
    """
    Tests the check for ignored alternative requests by the system in an inform by alternatives
    action.

    Args:
        simulator: Simulator object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    alt_request = UserAct(act_type=UserActionType.RequestAlternatives, slot=constraintA['slot'])
    user_actions = [UserAct(act_type=UserActionType.Request, slot=constraintA['slot']),
                    alt_request]
    sys_act = SysAct(act_type=SysActionType.InformByAlternatives, slot_values={
        primkey_constraint['slot']: [primkey_constraint['value']]})
    simulator.excluded_venues = [primkey_constraint['value']]
    res = simulator._check_system_ignored_request(user_actions, sys_act)
    assert type(res) == tuple
    assert len(res) == 2
    assert res[1] == [alt_request]
