import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.simulator.goal import Goal, Constraint
from utils import UserAct, UserActionType


def test_fill_agenda_on_initialization(agenda, goal, constraintA, constraintB):
    """
    Tests whether the agenda is emptied and filled with the constraints of the given goal on
    initialization.

    Args:
        agenda: Agenda object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    stored_action = UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                            value=constraintA['value'])
    goal_constraint = Constraint(slot=constraintB['slot'], value=constraintB['value'])
    agenda.stack = [stored_action]
    goal.constraints = [goal_constraint]
    agenda.init(goal)
    assert len(agenda.stack) == len(goal.constraints)
    assert stored_action not in agenda.stack
    assert any(item.slot == goal_constraint.slot and item.value == goal_constraint.value and
               item.type == UserActionType.Inform for item in agenda.stack)


def test_push_item(agenda, constraintA):
    """
    Tests whether pushing an item on the stack increases the stack size by one and inserts the
    item at the end of the stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    item = UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                   value=constraintA['value'])
    stack_size = len(agenda.stack)
    agenda.push(item)
    assert len(agenda.stack) == stack_size + 1
    assert agenda.stack[-1] == item


def test_push_several_items(agenda, constraintA, constraintB):
    """
    Tests whether pushing several items on the stack increases the stack size by the number of
    items and inserts the items in the same order at the end of the stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    items = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                     value=constraintA['value']),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value'])]
    stack_size = len(agenda.stack)
    agenda.push(items)
    assert len(agenda.stack) == stack_size + len(items)
    assert agenda.stack[-len(items):] == items


def test_get_actions_for_valid_num_actions(agenda, constraintA, constraintB):
    """
    Tests whether retrieving actions from the stack with a number of actions is smaller or equal
    to the stack size will return the last num_actions items from the stack in reversed order.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                     value=constraintA['value']),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value']),
             UserAct(act_type=UserActionType.Request, slot=constraintA['slot'],
                     value=constraintA['value'])]
    agenda.stack = stack.copy()
    num_actions = 2
    res = agenda.get_actions(num_actions=num_actions)
    assert len(res) == num_actions
    assert res == stack[-1:(len(stack)-(num_actions+1)):-1]


@pytest.mark.parametrize('num_actions', [3, -1])
def test_get_actions_for_invalid_num_actions(agenda, constraintA, constraintB, num_actions):
    """
    Tests whether retrieving actions from the stack with a number of actions that is less or
    greater than the stack size will return the complete stack in reversed order.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        num_actions (int): number of actions to be selected
    """
    stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                     value=constraintA['value']),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value'])]
    agenda.stack = stack.copy()
    res = agenda.get_actions(num_actions=num_actions)
    assert len(res) == len(stack)
    assert res == stack[::-1]


def test_keep_order_in_stack_on_clean(agenda, goal, constraintA, constraintB):
    """
    Tests whether cleaning the stack keeps the items in the same order.

    Args:
        agenda: Agenda object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                     value=constraintA['value']),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value'])]
    agenda.stack = stack.copy()
    goal.requests = {constraintA['slot']: None}
    agenda.clean(goal)
    assert len(agenda.stack) == len(stack)
    assert agenda.stack == stack


def test_remove_fulfilled_requests_on_clean(agenda, goal, constraintA, constraintB):
    """
    Tests whether cleaning the stack removes requests that are already fulfilled in the goal.

    Args:
        agenda: Agenda object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    fulfilled_request = UserAct(act_type=UserActionType.Request, slot=constraintA['slot'],
                     value=constraintA['value'])
    stack = [fulfilled_request, UserAct(act_type=UserActionType.Inform,
                                                 slot=constraintB['slot'],
                                                 value=constraintB['value'])]
    agenda.stack = stack.copy()
    goal.requests = {fulfilled_request.slot: fulfilled_request.value}
    agenda.clean(goal)
    assert len(agenda.stack) < len(stack)
    assert fulfilled_request not in agenda.stack


def test_remove_inconsistencies_on_clean(agenda, goal, constraintA, constraintB, constraintA_alt):
    """
    Tests whether cleaning the stack removes inconsistent inform constraints.

    Args:
        agenda: Agenda object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintA_alt (dict): as constraint A, but with an alternative value (given in
        conftest_<domain>.py)
    """
    inconsistent_item = UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                                value=constraintA['value'])
    stack = [inconsistent_item, UserAct(act_type=UserActionType.Confirm,
                                        slot=constraintB['slot'], value=constraintB['value'])]
    agenda.stack = stack.copy()
    goal.constraints = [Constraint(constraintA_alt['slot'], constraintA_alt['value'])]
    agenda.clean(goal)
    assert len(agenda.stack) < len(stack)
    assert all(item.slot != inconsistent_item.slot and item.value != inconsistent_item.value for
               item in agenda.stack)


def test_clear_stack(agenda, constraintA, constraintB):
    """
    Tests whether clearing the stack will remove all items from it.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                     value=constraintA['value']),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value'])]
    agenda.stack = stack.copy()
    agenda.clear()
    assert len(agenda.stack) == 0


def test_is_empty_on_empty_agenda(agenda):
    """
    Tests whether an empty agenda is recognized as empty.

    Args:
        agenda: Agenda object (given in conftest.py)
    """
    agenda.stack = []
    res = agenda.is_empty()
    assert res is True


def test_is_empty_on_nonempty_agenda(agenda, constraintA):
    """
    Tests whether an agenda with items is recognized as non-empty.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    agenda.stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                            value=constraintA['value'])]
    res = agenda.is_empty()
    assert res is False


def test_contains_action_of_type_with_matching_action(agenda, constraintA):
    """
    Tests whether asking for the existence of an action of specific type recognizes matching
    actions in the stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    agenda.stack  = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                             value='dontcare')]
    res = agenda.contains_action_of_type(act_type=UserActionType.Inform, consider_dontcare=True)
    assert res is True


def test_contains_action_of_type_without_matching_action(agenda, constraintA):
    """
    Tests whether asking for the existence of an action of specific type recognizes if no
    matching actions are in the stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    agenda.stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                            value='dontcare')]
    res = agenda.contains_action_of_type(act_type=UserActionType.Confirm)
    assert res is False


def test_contains_action_of_type_with_matching_action_without_considering_dontcare(agenda,
                                                                                   constraintA):
    """
    Tests whether asking for the existence of an action of specific type recognizes if no matching
    actions are in the stack that do not have a 'dontcare' value if specified.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    agenda.stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                            value='dontcare')]
    res = agenda.contains_action_of_type(act_type=UserActionType.Inform, consider_dontcare=False)
    assert res is False


def test_get_actions_of_type_with_matching_action(agenda, constraintA, constraintB):
    """
    Tests whether asking to get an action of specific type returns matching actions in the
    stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    act_type = UserActionType.Inform
    stack = [UserAct(act_type=act_type, slot=constraintA['slot'], value='dontcare'),
             UserAct(act_type=act_type, slot=constraintB['slot'], value=constraintB['value'])]
    agenda.stack = stack.copy()
    res = agenda.get_actions_of_type(act_type=act_type, consider_dontcare=True)
    assert list(res) == [act for act in stack if act.type == act_type]


def test_get_actions_of_type_without_matching_action(agenda, constraintA, constraintB):
    """
    Tests whether asking to get an action of specific type returns nothing if there are no matching
    actions in  the stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                     value=constraintA['value']),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value'])]
    agenda.stack = stack.copy()
    res = agenda.get_actions_of_type(act_type=UserActionType.Request, consider_dontcare=True)
    assert list(res) == []


def test_get_actions_of_type_with_matching_action_without_considering_dontcare(agenda, constraintA, constraintB):
    """
    Tests whether asking to get an action of specific type returns no matching actions in the
    stack that have the value 'dontcare' if specified.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    act_type = UserActionType.Inform
    stack = [UserAct(act_type=act_type, slot=constraintA['slot'], value='dontcare'),
             UserAct(act_type=act_type, slot=constraintB['slot'], value=constraintB['value'])]
    agenda.stack = stack.copy()
    res = agenda.get_actions_of_type(act_type=act_type, consider_dontcare=False)
    res = list(res)
    assert len(res) < len(stack)
    assert res == [act for act in stack if act.type == act_type and act.value != 'dontcare']


def test_remove_actions_of_type_with_matching_action(agenda, constraintA, constraintB):
    """
    Tests whether removing actions of a specific type will delete these actions if present in the stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    act_type = UserActionType.Inform
    stack = [UserAct(act_type=act_type, slot=constraintA['slot'],
                     value=constraintA['value']),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value'])]
    agenda.stack = stack.copy()
    agenda.remove_actions_of_type(act_type=act_type)
    assert len(agenda.stack) < len(stack)
    assert all(item.type != act_type for item in agenda.stack)


def test_remove_actions_of_type_without_matching_action(agenda, constraintA, constraintB):
    """
    Tests whether removing actions of a specific type will not change the stack if no matching
    actions are in it.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    stack = [UserAct(act_type=UserActionType.Inform, slot=constraintA['slot'],
                     value=constraintA['value']),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value'])]
    agenda.stack = stack.copy()
    agenda.remove_actions_of_type(act_type=UserActionType.Request)
    assert len(agenda.stack) == len(stack)
    assert agenda.stack == stack


def test_remove_actions_with_value(agenda, constraintA, constraintB):
    """
    Tests whether removing specific actions with stating their type, slot and value removes all
    matching actions from the stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    act_type = UserActionType.Inform
    slot = constraintA['slot']
    value = constraintA['value']
    stack = [UserAct(act_type=act_type, slot=slot, value=value),
             UserAct(act_type=UserActionType.Confirm, slot=constraintB['slot'],
                     value=constraintB['value'])]
    agenda.stack = stack.copy()
    agenda.remove_actions(act_type=act_type, slot=slot, value=value)
    assert len(agenda.stack) < len(stack)
    assert all(item.type != act_type and item.slot != slot and item.value != value for item in
               agenda.stack)


def test_remove_actions_without_value(agenda, constraintA, constraintB):
    """
    Tests whether removing specific actions with stating their type and slotremoves all
    matching actions from the stack, regardless of their value.

    Args:
        agenda: Agenda object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    act_type = UserActionType.Inform
    slot = constraintA['slot']
    stack = [UserAct(act_type=act_type, slot=slot, value=constraintA['value']),
             UserAct(act_type=act_type, slot=slot, value=constraintB['value'])]
    agenda.stack = stack.copy()
    agenda.remove_actions(act_type=act_type, slot=slot, value=None)
    assert len(agenda.stack) < len(stack)
    assert all(item.type != act_type and item.slot != slot for item in agenda.stack)


def test_fill_with_requests_with_name(agenda, goal, primkey_constraint, constraintA):
    """
    Tests whether filling the stack with requests will add user actions for each request in the
    goal to the stack. If the name should not be excluded, even requests for the primary key are
    added.

    Args:
        agenda: Agenda object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    agenda_size = len(agenda.stack)
    goal.requests = {primkey_constraint['slot']: None, constraintA['slot']: None}
    agenda.fill_with_requests(goal, exclude_name=False)
    assert len(agenda.stack) == agenda_size + len(goal.requests)
    assert any(item.type == UserActionType.Request and item.slot == primkey_constraint['slot']
               and  item.value is None for item in agenda.stack)
    assert any(item.type == UserActionType.Request and item.slot == constraintA['slot']
               and item.value is None for item in agenda.stack)


def test_fill_with_requests_without_name(agenda, goal, primkey_constraint, constraintA):
    """
    Tests whether filling the stack with requests will add user requests for each request in
    the
    goal to the stack. If the name should be excluded, no requests for the primary key are
    added.

    Args:
        agenda: Agenda object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    agenda_size = len(agenda.stack)
    goal.requests = {primkey_constraint['slot']: None, constraintA['slot']: None}
    agenda.fill_with_requests(goal, exclude_name=True)
    assert len(agenda.stack) > agenda_size
    assert all(item.type == UserActionType.Request and item.slot != primkey_constraint['slot']
               for item in agenda.stack)
    assert any(item.type == UserActionType.Request and item.slot == constraintA['slot']
               and item.value is None for item in agenda.stack)


def test_fill_with_constraints(agenda, goal, constraintA, constraintB):
    """
    Tests whether filling the stack with constraints will add all constraints from the goal as
    user actions of type inform to the stack.

    Args:
        agenda: Agenda object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    goal.constraints = [Constraint(slot=constraintA['slot'], value=constraintA['value']),
                        Constraint(slot=constraintB['slot'], value=constraintB['value'])]
    agenda_size = len(agenda.stack)
    agenda.fill_with_constraints(goal)
    assert len(agenda.stack) == agenda_size + len(goal.constraints)
    assert any(item.type == UserActionType.Inform and item.slot == constraintA['slot'] and
               item.value == constraintA['value'] for item in agenda.stack)
    assert any(item.type == UserActionType.Inform and item.slot == constraintB['slot'] and
               item.value == constraintB['value'] for item in agenda.stack)


