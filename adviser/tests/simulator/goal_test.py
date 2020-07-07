import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.simulator.goal import Constraint


def test_init_random_goal_without_parameters(goal):
    """
    Tests the initialization of a random goal without providing parameters.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.init(random_goal=True)
    assert goal.constraints != []
    assert goal.requests != {}


def test_init_random_goal_with_parameters(goal):
    """
    Tests the initialization of a random goal with given parameters.

    Args:
        goal: Goal object (given in conftest.py)
    """
    min_constraints = 1
    max_constraints = 3
    min_requests = 1
    max_requests = 4
    goal.parameters = {'MinVenues': 3, 'MinConstraints': min_constraints,
                       'MaxConstraints': max_constraints, 'Reachable': 0.7,
                       'MinRequests': min_requests, 'MaxRequests': max_requests}
    goal.init(random_goal=True, constraints=None, requests=None)
    print(goal.parameters)
    assert goal.constraints != []
    assert goal.requests != {}
    assert len(goal.constraints) >= min_constraints
    assert len(goal.constraints) <= max_constraints
    assert len(goal.requests) >= min_requests
    assert len(goal.requests) <= max_requests + 1 # primary key slot is added separately

@pytest.mark.parametrize('constraints', [
    lambda x, y: [(x['slot'], x['value']), (y['slot'], y['value'])], # as list of tuples
    lambda x, y: [Constraint(x['slot'], x['value']), Constraint(y['slot'], y['value'])], # as list of Constraints
    lambda x, y : {x['slot']: x['value'], y['slot']: y['value']} # as dict
])
@pytest.mark.parametrize('requests', [
    lambda x, y : [x['slot'], y['slot']], # as list of strings (slots)
    lambda x, y: {x['slot']: None, y['slot']: None} # as dict
])
def test_init_goal_from_parameters(goal, constraints, requests, constraintA, constraintB):
    """
    Tests the initialization of a goal from constraints and requests given as parameters. Tests
    several options how constraints and requests can be provided (e.g. as list of tuples).

    Args:
        goal: Goal object (given in conftest.py)
        constraints: lambda function to initialize the constraints parameter
        requests: lambda function to initialize the requests parameter
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    goal.init(constraints=constraints(constraintA, constraintB),
              requests=requests(constraintA, constraintB))
    assert goal.constraints != []
    assert goal.requests != {}


def test_init_goal_from_parameters_with_invalid_constraints(goal):
    """
    Tests whether initializing a goal with constrainst givne in an invalid form fails.

    Args:
        goal: Goal object (given in conftest.py)
    """
    with pytest.raises(ValueError):
        goal.init(constraints='foo', requests='bar')


def test_reset_requests(goal):
    """
    Tests whether resetting the goal removes all request values.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.requests = {'foo': 'bar'}
    goal.reset()
    assert all(value == None for value in goal.requests.values())


def test_goal_is_fulfilled(goal):
    """
    Tests whether a goal can be recognized as fulfilled.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.requests = {'foo': 'bar', 'bar': 'foo'}
    is_fulfilled = goal.is_fulfilled()
    assert is_fulfilled is True


def test_fulfill_request(goal):
    """
    Tests whether fulfilling a request updates the value of the corresponding slot in the
    requests dict.

    Args:
        goal: Goal object (given in conftest.py)
    """
    slot = 'foo'
    value = 'bar'
    goal.requests = {slot: None}
    goal.fulfill_request(slot, value)
    assert slot in goal.requests
    assert goal.requests[slot] == value


def test_is_inconsistent_constraint_for_inconsistent_constraint(goal):
    """
    Tests whether an inconsistent constraint is recognized as such by the base function.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'bar')]
    is_inconsistent = goal.is_inconsistent_constraint(Constraint('foo', 'baz'))
    assert is_inconsistent is True


def test_is_inconsistent_constraint_for_consistent_constraint(goal):
    """
    Tests whether a consistent constraint is recognized as such by the base function.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'bar')]
    is_inconsistent = goal.is_inconsistent_constraint(Constraint('foo', 'bar'))
    assert is_inconsistent is False


def test_is_inconsistent_constraint_for_dontcare_constraint(goal):
    """
    Tests whether an inconsistent constraint is not recognized as such by the base function if
    the value for the corresponding goal constraint is 'dontcare'.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'dontcare')]
    is_inconsistent = goal.is_inconsistent_constraint(Constraint('foo', 'bar'))
    assert is_inconsistent is False


def test_is_inconsistent_constraint_for_unknown_constraint(goal):
    """
    Tests whether an unknown constraint is not recognized as inconsistent by the base function.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'bar')]
    is_inconsistent = goal.is_inconsistent_constraint_strict(Constraint('bar', 'foo'))
    assert is_inconsistent is True


def test_is_inconsistent_constraint_strict_for_inconsistent_constraint(goal):
    """
    Tests whether an inconsistent constraint is recognized as such by the strict function.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'bar')]
    is_inconsistent = goal.is_inconsistent_constraint_strict(Constraint('foo', 'baz'))
    assert is_inconsistent is True


def test_is_inconsistent_constraint_strict_for_consistent_constraint(goal):
    """
    Tests whether a consistent constraint is recognized as such by the strict function.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'bar')]
    is_inconsistent = goal.is_inconsistent_constraint_strict(Constraint('foo', 'bar'))
    assert is_inconsistent is False


def test_is_inconsistent_constraint_strict_for_dontcare_constraint(goal):
    """
    Tests whether an inconsistent constraint is recognized as such by the strict function even if
    the value for the corresponding goal constraint is 'dontcare'.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'dontcare')]
    is_inconsistent = goal.is_inconsistent_constraint_strict(Constraint('foo', 'bar'))
    assert is_inconsistent is True


def test_is_inconsistent_constraint_strict_for_unknown_constraint(goal):
    """
    Tests whether an unknown constraint is not recognized as inconsistent by the strict function.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'bar')]
    is_inconsistent = goal.is_inconsistent_constraint_strict(Constraint('bar', 'foo'))
    assert is_inconsistent is True


def test_get_known_constraint(goal):
    """
    Tests whether the get_constraint function returns the corresponding value if the goal has a
    constraint with that slot.

    Args:
        goal: Goal object (given in conftest.py)
    """
    slot = 'foo'
    value = 'bar'
    goal.constraints = [Constraint(slot, value)]
    constraint_value = goal.get_constraint(slot)
    assert constraint_value == value


def test_get_unknown_constraint(goal):
    """
    Tests whether the get_constraint function returns a 'dontcare' if the goal has no constraint
    with that slot.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'bar')]
    constraint_value = goal.get_constraint('baz')
    assert constraint_value == 'dontcare'


def test_update_known_constraint(goal):
    """
    Tests whether updating a constraint works correctly if the constraint is actually present in
    the goal's constraints.

    Args:
        goal: Goal object (given in conftest.py)
    """
    slot = 'foo'
    value = 'baz'
    goal.constraints = [Constraint(slot, 'bar')]
    update_successful = goal.update_constraint(slot, value)
    matching_constraint = next(constraint for constraint in goal.constraints
                               if constraint.slot == slot)
    assert update_successful is True
    assert matching_constraint.value == value


def test_update_unknown_constraint(goal):
    """
    Tests whether updating a constraint is skipped if the constraint is not present in the goal's constraints.

    Args:
        goal: Goal object (given in conftest.py)
    """
    goal.constraints = [Constraint('foo', 'bar')]
    update_successful = goal.update_constraint('bar', 'foo')
    assert update_successful is False