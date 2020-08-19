import os
import sys


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.stats.evaluation import PolicyEvaluator


def create_evaluator(domain):
    """
    Creates a PolicyEvaluator object and assigns values to the class variables that are different
    from the initial values.

    Args:
        domain: Domain object (given in conftest.py)
    """
    evaluator = PolicyEvaluator(domain=domain)
    evaluator.dialog_start()
    evaluator.epoch = 5
    evaluator.total_train_dialogs = 20
    evaluator.total_eval_dialogs = 10
    evaluator.epoch_train_dialogs = 3
    evaluator.epoch_eval_dialogs = 2
    evaluator.train_rewards = [0, 20, 0]
    evaluator.eval_rewards = [0, 0]
    evaluator.train_success = [0, 1, 0]
    evaluator.eval_success = [0, 0]
    evaluator.train_turns = [0, 2, 4]
    evaluator.eval_turns = [1, 3]
    evaluator.is_training = False
    return evaluator


def test_evaluate_turn_decreases_reward(domain):
    """
    Tests whether the dialog reward is decreased by the turn reward when evaluating a turn.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    turn_reward = -10
    evaluator = PolicyEvaluator(domain=domain, turn_reward=turn_reward)
    evaluator.dialog_start()
    intial_reward = evaluator.dialog_reward
    res = evaluator.evaluate_turn()
    assert isinstance(res, dict)
    assert 'sys_turn_over' in res
    assert res['sys_turn_over'] is True
    assert evaluator.dialog_reward == intial_reward + turn_reward


def test_evaluate_turn_increases_turns(domain):
    """
    Tests whether the number of dialog turns is increased by one when evaluating a turn.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    evaluator = create_evaluator(domain)
    intial_turns = evaluator.dialog_turns
    res = evaluator.evaluate_turn()
    assert isinstance(res, dict)
    assert 'sys_turn_over' in res
    assert res['sys_turn_over'] is True
    assert evaluator.dialog_turns == intial_turns + 1


def test_dialog_start_resets_reward(domain):
    """
    Tests whether starting a dialog resets the dialog reward.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    evaluator = create_evaluator(domain)
    evaluator.dialog_reward = 20
    evaluator.dialog_start()
    assert evaluator.dialog_reward == 0


def test_dialog_start_resets_turns(domain):
    """
    Tests whether starting a dialog resets the number of dialog turns.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    evaluator = create_evaluator(domain)
    evaluator.dialog_turns = 20
    evaluator.dialog_start()
    assert evaluator.dialog_turns == 0


def test_train(domain):
    """
    Tests whether train activates the training modus.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    evaluator = create_evaluator(domain)
    evaluator.is_training = False
    evaluator.train()
    assert evaluator.is_training is True


def test_eval(domain):
    """
    Test whether eval deactivates the training modus.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    evaluator = create_evaluator(domain)
    evaluator.is_training = True
    evaluator.eval()
    assert evaluator.is_training is False


def test_end_dialog_in_training_updates_train_statistics(domain, goal, primkey_constraint):
    """
    Tests whether the training statistics are updated when ending a dialog during training.

    Args:
        domain:  Domain object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    goal.requests['name'] = primkey_constraint['value']
    evaluator = create_evaluator(domain)
    evaluator.is_training = True
    previous_total_train_dialogs = evaluator.total_train_dialogs
    previous_epoch_train_dialogs = evaluator.epoch_train_dialogs
    previous_dialog_reward = evaluator.dialog_reward
    previous_train_rewards = evaluator.train_rewards[:]
    previous_train_success = evaluator.train_success[:]
    previous_train_turns = evaluator.train_turns[:]
    res = evaluator.end_dialog(goal)
    assert isinstance(res, dict)
    assert 'dialog_end' in res
    assert res['dialog_end'] is True
    assert evaluator.total_train_dialogs == previous_total_train_dialogs + 1
    assert evaluator.epoch_train_dialogs == previous_epoch_train_dialogs + 1
    assert evaluator.dialog_reward >= previous_dialog_reward
    assert len(evaluator.train_rewards) == len(previous_train_rewards) + 1
    assert evaluator.train_rewards[-1] == evaluator.dialog_reward
    assert len(evaluator.train_success) == len(previous_train_success) + 1
    assert evaluator.train_success[-1] in (0, 1)
    assert len(evaluator.train_turns) == len(previous_train_turns) + 1
    assert evaluator.train_turns[-1] == evaluator.dialog_turns


def test_end_dialog_in_evaluation_updates_eval_statistics(domain, goal, primkey_constraint):
    """
    Tests whether the evaluation statistics are updated when ending a dialog during evaluation.

    Args:
        domain:  Domain object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    goal.requests['name'] = primkey_constraint['value']
    evaluator = create_evaluator(domain)
    evaluator.is_training = False
    previous_total_eval_dialogs = evaluator.total_eval_dialogs
    previous_epoch_eval_dialogs = evaluator.epoch_eval_dialogs
    previous_dialog_reward = evaluator.dialog_reward
    previous_eval_rewards = evaluator.eval_rewards[:]
    previous_eval_success = evaluator.eval_success[:]
    previous_eval_turns = evaluator.eval_turns[:]
    res = evaluator.end_dialog(goal)
    assert isinstance(res, dict)
    assert 'dialog_end' in res
    assert res['dialog_end'] is True
    assert evaluator.total_eval_dialogs == previous_total_eval_dialogs + 1
    assert evaluator.epoch_eval_dialogs == previous_epoch_eval_dialogs + 1
    assert evaluator.dialog_reward >= previous_dialog_reward
    assert len(evaluator.eval_rewards) == len(previous_eval_rewards) + 1
    assert evaluator.eval_rewards[-1] == evaluator.dialog_reward
    assert len(evaluator.eval_success) == len(previous_eval_success) + 1
    assert evaluator.eval_success[-1] in (0, 1)
    assert len(evaluator.eval_turns) == len(previous_eval_turns) + 1
    assert evaluator.eval_turns[-1] == evaluator.dialog_turns


def test_end_dialog_in_training_ignores_eval_statistics(domain, goal, primkey_constraint):
    """
    Tests whether the evaluation statistics are left unchanged when ending a dialog during training.

    Args:
        domain:  Domain object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    goal.requests['name'] = primkey_constraint['value']
    evaluator = create_evaluator(domain)
    evaluator.is_training = True
    previous_total_eval_dialogs = evaluator.total_eval_dialogs
    previous_epoch_eval_dialogs = evaluator.epoch_eval_dialogs
    previous_eval_rewards = evaluator.eval_rewards[:]
    previous_eval_success = evaluator.eval_success[:]
    previous_eval_turns = evaluator.eval_turns[:]
    evaluator.end_dialog(goal)
    assert evaluator.total_eval_dialogs == previous_total_eval_dialogs
    assert evaluator.epoch_eval_dialogs == previous_epoch_eval_dialogs
    assert evaluator.eval_rewards == previous_eval_rewards
    assert evaluator.eval_success == previous_eval_success
    assert evaluator.eval_turns == previous_eval_turns


def test_end_dialog_in_evaluation_ignores_train_statistics(domain, goal, primkey_constraint):
    """
    Tests whether the training statistics are left unchanged when ending a dialog during evaluation.

    Args:
        domain:  Domain object (given in conftest.py)
        goal: Goal object (given in conftest.py)
        primkey_constraint (dict): slot-value pair for a primary key constraint (given in
        conftest_<domain>.py)
    """
    goal.requests['name'] = primkey_constraint['value']
    evaluator = create_evaluator(domain)
    evaluator.is_training = False
    previous_total_train_dialogs = evaluator.total_train_dialogs
    previous_epoch_train_dialogs = evaluator.epoch_train_dialogs
    previous_train_rewards = evaluator.train_rewards[:]
    previous_train_success = evaluator.train_success[:]
    previous_train_turns = evaluator.train_turns[:]
    evaluator.end_dialog(goal)
    assert evaluator.total_train_dialogs == previous_total_train_dialogs
    assert evaluator.epoch_train_dialogs == previous_epoch_train_dialogs
    assert evaluator.train_rewards == previous_train_rewards
    assert evaluator.train_success == previous_train_success
    assert evaluator.train_turns == previous_train_turns


def test_end_dialog_with_real_user_interaction(domain):
    """
    Tests whether ending a dialog with a real user interaction (instead of a simulator) does not
    change the class variables (e.g. the dialog reward) because there is no need to evaluate
    anything then.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    evaluator = create_evaluator(domain)
    previous_dialog_reward = evaluator.dialog_reward
    res = evaluator.end_dialog(sim_goal=None)
    assert isinstance(res, dict)
    assert 'dialog_end' in res
    assert res['dialog_end'] is True
    assert evaluator.dialog_reward == previous_dialog_reward


def test_start_epoch_resets_variables(domain):
    """
    Tests whether starting an epoch resets the class variables.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    epoch = 10
    evaluator = create_evaluator(domain)
    evaluator.epoch = epoch
    evaluator.start_epoch()
    assert evaluator.epoch_train_dialogs == 0
    assert evaluator.epoch_eval_dialogs == 0
    assert evaluator.train_rewards == []
    assert evaluator.eval_rewards == []
    assert evaluator.train_success == []
    assert evaluator.eval_success == []
    assert evaluator.train_turns == []
    assert evaluator.eval_turns == []
    assert evaluator.epoch == epoch + 1



def test_end_epoch_in_training_returns_train_statistics(domain):
    """
    Tests whether ending an epoch during training returns the training statistics.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    evaluator = create_evaluator(domain)
    evaluator.is_training = True
    res = evaluator.end_epoch()
    assert isinstance(res, dict)
    assert 'num_dialogs' in res
    assert res['num_dialogs'] == evaluator.epoch_train_dialogs
    assert 'turns' in res
    assert res['turns'] == sum(evaluator.train_turns) / evaluator.epoch_train_dialogs
    assert 'success' in res
    assert res['success'] == float(sum(evaluator.train_success)) / evaluator.epoch_train_dialogs
    assert 'reward' in res
    assert res['reward'] == float(sum(evaluator.train_rewards)) / evaluator.epoch_train_dialogs



def test_end_epoch_in_evaluation_returns_eval_statistics(domain):
    """
    Tests whether ending an epoch during evaluation returns the evaluation statistics.

    Args:
        domain:  Domain object (given in conftest.py)
    """
    evaluator = create_evaluator(domain)
    evaluator.is_training = False
    res = evaluator.end_epoch()
    assert isinstance(res, dict)
    assert 'num_dialogs' in res
    assert res['num_dialogs'] == evaluator.epoch_eval_dialogs
    assert 'turns' in res
    assert res['turns'] == sum(evaluator.eval_turns) / evaluator.epoch_eval_dialogs
    assert 'success' in res
    assert res['success'] == float(sum(evaluator.eval_success)) / evaluator.epoch_eval_dialogs
    assert 'reward' in res
    assert res['reward'] == float(sum(evaluator.eval_rewards)) / evaluator.epoch_eval_dialogs
