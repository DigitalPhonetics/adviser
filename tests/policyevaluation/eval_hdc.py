###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
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

import os
import sys
import random

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(get_root_dir())

import utils.logger as log
log.logfile_base_name = 'eval_hdc_imsmpdules'
log.console_level = log.LogLevel.RESULTS
log.file_level = log.LogLevel.DIALOGS
logger = log.DiasysLogger()
from modules.policy.rl.experience_buffer import UniformBuffer, NaivePrioritizedBuffer

from dialogsystem import DialogSystem
from modules.bst import HandcraftedBST
from modules.policy import HandcraftedPolicy
from modules.policy.evaluation import PolicyEvaluator
from modules.simulator import HandcraftedUserSimulator
from utils.domain.jsonlookupdomain import JSONLookupDomain

from utils import common

if __name__ == "__main__":
    common.init_random()

    NUM_TEST_SEEDS = 10
    EVAL_EPISODES = 500
    MAX_TURNS = -1

    # get #num_test_seeds random seeds
    random_seeds = []
    for i in range(NUM_TEST_SEEDS):
        random_seeds.append(random.randint(0, 2**32-1))

    results = {}
    for seed in random_seeds:
        common.init_once = False
        common.init_random(seed=seed)    
        domain = JSONLookupDomain('ImsCourses')
        bst = HandcraftedBST(domain=domain)
        user_simulator = HandcraftedUserSimulator(domain=domain)
        policy = None

        policy= HandcraftedPolicy(domain=domain)
        evaluator = PolicyEvaluator(domain=domain, use_tensorboard=True, 
                                    experiment_name='eval_hdc_imscourses')
        ds = DialogSystem(policy,
                        user_simulator,
                        bst,
                        evaluator
                            )

        ds.eval()
        evaluator.start_epoch()
        for episode in range(EVAL_EPISODES):
            ds.run_dialog(max_length=MAX_TURNS)
        evaluator.end_epoch()

        results[seed] = {}
        results[seed]['eval_dialogs'] = evaluator.epoch_eval_dialogs
        results[seed]['eval_avg_turns'] = sum(evaluator.eval_turns) / evaluator.epoch_eval_dialogs
        results[seed]['eval_avg_success'] = sum(evaluator.eval_success) / evaluator.epoch_eval_dialogs
        results[seed]['eval_avg_reward'] = sum(evaluator.eval_rewards) / evaluator.epoch_eval_dialogs

    logger.result("")
    logger.result("###################################################")
    logger.result("")
    logger.result(" ### Eval with " + str(NUM_TEST_SEEDS) + " random seeds ###")
    logger.result(str(random_seeds))
    logger.result("# Num Dialogs " + str(EVAL_EPISODES))
    logger.result("# Avg Turns " + str(sum([results[seed]['eval_avg_turns'] for seed in random_seeds]) / NUM_TEST_SEEDS))
    logger.result("# Avg Success " + str(sum([results[seed]['eval_avg_success'] for seed in random_seeds]) / NUM_TEST_SEEDS))
    logger.result("# Avg Reward " + str(sum([results[seed]['eval_avg_reward'] for seed in random_seeds]) / NUM_TEST_SEEDS))

        