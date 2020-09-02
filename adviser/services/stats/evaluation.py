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


from services.service import Service, PublishSubscribe
from services.simulator.goal import Goal
from utils.domain.domain import Domain
from utils.logger import DiasysLogger
from utils.sysact import SysAct


class ObjectiveReachedEvaluator(object):
    """ Evaluate single turns and complete dialog.

        This class assigns a negative reward to each turn.
        In case the user's goal could be satisfied (meaning a matching database
        entry was found), a large final reward is returned.

        Only needed when training against a simulator.
    """

    def __init__(self, domain: Domain, turn_reward=-1, success_reward=20,
                 logger: DiasysLogger = DiasysLogger()):
        assert turn_reward <= 0.0, 'the turn reward should be negative'
        self.domain = domain
        self.turn_reward = turn_reward
        self.success_reward = success_reward
        self.logger = logger

    def get_turn_reward(self):
        """ 
        Get the reward for one turn
        
        Returns:
            (int): the reward for the given turn
        """
        return self.turn_reward

    def get_final_reward(self, sim_goal: Goal, logging=True):
        """ 
        Check whether the user's goal was completed.

        Args:
            sim_goal (Goal): the simulation's goal
            logging (bool): whether or not the evaluation results should be logged

        Returns:
            float: Reward - the final reward (0 (unsuccessful) or 20 (successful))
            bool: Success
        """

        requests = sim_goal.requests
        constraints = sim_goal.constraints  # list of constraints
        # self.logger.dialog_turn("User Goal > " + str(sim_goal.constraints))

        if None in requests.values() or requests['name'] == 'none':
            if logging:
                self.logger.dialog_turn("Fail with user requests \n{}".format(requests))
            return 0.0, False
            # TODO think about this more? if goals not satisfiable,
            # should system take the blame? not fair

        # print(requests['name'])
        db_matches = self.domain.find_info_about_entity(
            entity_id=requests['name'],
            requested_slots=[constraint.slot for constraint in constraints])
        if db_matches:
            match = db_matches[0]
            for const in constraints:
                if const.value != match[const.slot] and const.value != 'dontcare':
                    if logging:
                        self.logger.dialog_turn("Fail with user requests \n{}".format(requests))
                    return 0.0, False
            if logging:
                self.logger.dialog_turn("Success with user requests \n{}".format(requests))
            return 20.0, True

        if logging:
            self.logger.dialog_turn("Fail with user requests \n{}".format(requests))
        return 0.0, False


class PolicyEvaluator(Service):
    """ Policy evaluation module

    Plug this module into the dialog graph (somewhere *after* the policy),
    and policy metrics like success rate and reward will be recorded.

    """

    def __init__(self, domain: Domain, subgraph: dict = None, use_tensorboard=False,
                 experiment_name: str = '', turn_reward=-1, success_reward=20,
                 logger: DiasysLogger = DiasysLogger(), summary_writer=None):
        """
        Keyword Arguments:
            use_tensorboard {bool} -- [If true, metrics will be written to
                                       tensorboard in a *runs* directory]
                                       (default: {False})
            experiment_name {str} -- [Name suffix for the log files]
                                      (default: {''})
            turn_reward {float} -- [Reward for one turn - usually negative to
                                    penalize dialog length] (default: {-1})
            success_reward {float} -- [Reward of the final transition if the
                                       dialog goal was reached] (default: {20})
        """
        super(PolicyEvaluator, self).__init__(domain)
        self.logger = logger
        self.epoch = 0
        self.evaluator = ObjectiveReachedEvaluator(
            domain, turn_reward=turn_reward, success_reward=success_reward, logger=logger)

        self.writer = summary_writer

        self.total_train_dialogs = 0
        self.total_eval_dialogs = 0

        self.epoch_train_dialogs = 0
        self.epoch_eval_dialogs = 0
        self.train_rewards = []
        self.eval_rewards = []
        self.train_success = []
        self.eval_success = []
        self.train_turns = []
        self.eval_turns = []
        self.is_training = False

    @PublishSubscribe(sub_topics=['sys_act'], pub_topics=["sys_turn_over"])
    def evaluate_turn(self, sys_act: SysAct = None):
        """
            Evaluates the reward for a given turn

            Args:
                sys_act (SysAct): the system action

            Returns:
                (bool): A signal representing the end of a complete dialog turn
        """
        self.dialog_reward += self.evaluator.get_turn_reward()
        self.dialog_turns += 1

        return {"sys_turn_over": True}

    def dialog_start(self, dialog_start=False):
        """
            Clears the state of the evaluator in preparation to start a new dialog
        """
        self.dialog_reward = 0.0
        self.dialog_turns = 0

    def train(self):
        """
            sets the evaluator in train mode
        """
        self.is_training = True

    def eval(self):
        """
            sets teh evaluator in eval mode
        """
        self.is_training = False

    @PublishSubscribe(sub_topics=["sim_goal"], pub_topics=["dialog_end"])
    def end_dialog(self, sim_goal: Goal):
        """
            Method for handling the end of a dialog; calculates the the final reward.

            Args:
                sim_goal (Goal): the simulation goal to evaluate against

            Returns:
                (dict): a dictionary where the key is "dialog_end" and the value is true
        """
        if self.is_training:
            self.total_train_dialogs += 1
            self.epoch_train_dialogs += 1
        else:
            self.total_eval_dialogs += 1
            self.epoch_eval_dialogs += 1

        if sim_goal is None:
            # real user interaction, no simulator - don't have to evaluate
            # anything, just reset counters
            return {"dialog_end": True}

        final_reward, success = self.evaluator.get_final_reward(sim_goal)
        self.dialog_reward += final_reward

        if self.is_training:
            self.train_rewards.append(self.dialog_reward)
            self.train_success.append(int(success))
            self.train_turns.append(self.dialog_turns)
            if self.writer is not None:
                self.writer.add_scalar('train/episode_reward', self.dialog_reward,
                                       self.total_train_dialogs)
        else:
            self.eval_rewards.append(self.dialog_reward)
            self.eval_success.append(int(success))
            self.eval_turns.append(self.dialog_turns)
            if self.writer is not None:
                self.writer.add_scalar('eval/episode_reward', self.dialog_reward,
                                       self.total_eval_dialogs)

        return {"dialog_end": True}

    def start_epoch(self):
        """
        Handles resetting variables between epochs
        """

        # global statistics
        self.epoch_train_dialogs = 0
        self.epoch_eval_dialogs = 0
        self.train_rewards = []
        self.eval_rewards = []
        self.train_success = []
        self.eval_success = []
        self.train_turns = []
        self.eval_turns = []
        self.epoch += 1

        self.logger.info("###\n### EPOCH" + str(self.epoch) + " ###\n###")

    def end_epoch(self):
        """
        Handles calculating statistics at the end of an epoch        
        """
        
        if self.logger:
            if self.epoch_train_dialogs > 0:
                self.logger.result(" ### Train ###")
                self.logger.result("# Num Dialogs " + str(self.epoch_train_dialogs))
                self.logger.result("# Avg Turns " + str(sum(self.train_turns) / self.epoch_train_dialogs))
                self.logger.result("# Avg Success " + str(sum(self.train_success) / self.epoch_train_dialogs))
                self.logger.result("# Avg Reward " + str(sum(self.train_rewards) / self.epoch_train_dialogs))
            if self.epoch_eval_dialogs > 0:
                self.logger.result(" ### Eval ###")
                self.logger.result("# Num Dialogs " + str(self.epoch_eval_dialogs))
                self.logger.result("# Avg Turns " + str(sum(self.eval_turns) / self.epoch_eval_dialogs))
                self.logger.result("# Avg Success " + str(sum(self.eval_success) / self.epoch_eval_dialogs))
                self.logger.result("# Avg Reward " + str(sum(self.eval_rewards) / self.epoch_eval_dialogs))

        if self.is_training:
            return {'num_dialogs': self.epoch_train_dialogs,
                    'turns': sum(self.train_turns) / self.epoch_train_dialogs,
                    'success': float(sum(self.train_success)) / self.epoch_train_dialogs,
                    'reward': float(sum(self.train_rewards)) / self.epoch_train_dialogs}
        else:
            return {'num_dialogs': self.epoch_eval_dialogs,
                    'turns': sum(self.eval_turns) / self.epoch_eval_dialogs,
                    'success': float(sum(self.eval_success)) / self.epoch_eval_dialogs,
                    'reward': float(sum(self.eval_rewards)) / self.epoch_eval_dialogs}
