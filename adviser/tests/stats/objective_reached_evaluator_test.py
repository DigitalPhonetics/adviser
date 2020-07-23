import os
import sys

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
import pytest
from services.stats.evaluation import ObjectiveReachedEvaluator
from services.simulator.goal import Constraint


def test_get_turn_reward(domain):
    """
    Tests whether retrieving the turn reward returns the correct value.

    Args:
        domain: Domain object (given in conftest.py)
    """
    turn_reward = -10
    evaluator = ObjectiveReachedEvaluator(domain, turn_reward=turn_reward)
    res = evaluator.get_turn_reward()
    assert res is not None
    assert res == turn_reward


@pytest.mark.parametrize('requests', [{None: None, 'name': None}, {'name': 'none'}])
def test_get_final_reward_with_none_requests(domain, goal, requests):
    """
    Tests whether a goal containing a None-slot in the requests or the entity name 'none' is
    seen as uncompleted / unsuccessful (reward: 0)

    Args:
        domain: Domain object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        requests (dict): ill-formed request dictionaries
    """
    evaluator = ObjectiveReachedEvaluator(domain)
    goal.requests = requests
    reward, success = evaluator.get_final_reward(goal)
    assert reward == 0.0
    assert success is False


def test_get_final_reward_without_match_in_db(domain, goal):
    """
    Tests whether a goal containing a requested entity name that does not match any entity in
    the database is seen as uncompleted / unsuccessful (reward: 0).

    Args:
        domain: Domain object (given in conftest.py)
        goal: Goal object (given in conftest.py)
    """
    evaluator = ObjectiveReachedEvaluator(domain)
    goal.requests = {'name': 'foo'}
    reward, success = evaluator.get_final_reward(goal)
    assert reward == 0.0
    assert success is False


def test_get_final_reward_with_false_db_match(domain, goal, constraint_with_no_match,
                                              primkey_constraint):
    """
    Tests whether a goal containing a requested entity name that does not match the specified
    constraints is seen as uncompleted / unsuccessful (reward: 0).

    Args:
        domain: Domain object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        constraint_with_no_match (dict): slot-value pair for constraint without matches in
        the domain (given in conftest_<domain>.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    evaluator = ObjectiveReachedEvaluator(domain)
    goal.constraints = [Constraint(constraint_with_no_match['slot'],
                                   constraint_with_no_match['value'])]
    goal.requests = {'name': primkey_constraint['value']}
    reward, success = evaluator.get_final_reward(goal)
    assert reward == 0.0
    assert success is False

def test_get_final_reward_for_sucess(domain, goal, entryA):
    """
    Tests whether a goal containing a requested entity name that matches all specified
    constraints is seen as successful (reward: 20).

    Args:
        domain: Domain object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    evaluator = ObjectiveReachedEvaluator(domain)
    slots = set(domain.get_system_requestable_slots()) - {'name'}
    goal.requests = {'name': entryA['name']}
    goal.constraints = [Constraint(slot, entryA[slot]) for slot in slots]
    reward, success = evaluator.get_final_reward(goal)
    assert reward == 20.0
    assert success is True