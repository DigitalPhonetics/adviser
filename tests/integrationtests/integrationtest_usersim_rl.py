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
from statistics import mean

def get_root_dir():
    head_location = os.path.realpath(os.curdir)
    end = head_location.find('adviser')
    return os.path.join(head_location[:end], 'adviser')
sys.path.append(get_root_dir())


from modules.policy.rl.experience_buffer import UniformBuffer, NaivePrioritizedBuffer

from dialogsystem import DialogSystem
from modules.bst import HandcraftedBST
from modules.simulator import HandcraftedUserSimulator, SimpleNoise
from modules.policy import HandcraftedPolicy, DQNPolicy
from modules.policy.evaluation import PolicyEvaluator
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils import DiasysLogger, LogLevel


from utils import common

if __name__ == "__main__":
    logger = DiasysLogger(console_log_lvl=LogLevel.RESULTS, file_log_lvl=LogLevel.DIALOGS)
    
    turns = {}
    success = {}
    for domain in [JSONLookupDomain(domain_str) for domain_str in ['ImsCourses']]:
        turns[domain.name] = []
        success[domain.name] = []
        for i in range(5):
            common.init_random() # add seed here if wanted

            TRAIN_EPOCHS = 10
            TRAIN_EPISODES = 1000
            EVAL_EPISODES = 1000
            MAX_TURNS = 25

            bst = HandcraftedBST(domain=domain, logger=logger)
            user = HandcraftedUserSimulator(domain, logger=logger)
            noise = SimpleNoise(domain=domain, train_error_rate=0.30, test_error_rate=0.30, logger=logger)
            policy = DQNPolicy(domain=domain, lr=0.0001, eps_start=0.3, gradient_clipping=5.0, buffer_cls=NaivePrioritizedBuffer, replay_buffer_size=8192, shared_layer_sizes=[256], train_dialogs=TRAIN_EPISODES, target_update_rate=3, training_frequency=2, logger=logger)
            # policy = HandcraftedPolicy(domain=domain, logger=logger)
            evaluator = PolicyEvaluator(domain=domain, use_tensorboard=True, experiment_name=f'policy#{type(policy)}_epochs#{TRAIN_EPOCHS}_domain#{domain.name}_run#{i}', logger=logger)
            ds = DialogSystem(policy,
                            user,
                            # noise,
                            bst,
                            evaluator)

            for i in range(TRAIN_EPOCHS):
                ds.train()
                evaluator.start_epoch()
                for episode in range(TRAIN_EPISODES):
                    ds.run_dialog(max_length=MAX_TURNS)
                    # input()
                evaluator.end_epoch()
                #policy.buffer.print_contents(10000)
                ds.num_dialogs = 0
                ds.eval()
                evaluator.start_epoch()
                for episode in range(EVAL_EPISODES):
                    ds.run_dialog(max_length=MAX_TURNS)
                results = evaluator.end_epoch()
                ds.num_dialogs = 0 # IMPORTANT for epsilon scheduler in dqnpolicy

                policy.save()

            turns[domain.name].append(results['turns'])
            success[domain.name].append(results['success'])
    print("Success: ", {domain: mean(values) for domain, values in success.items()})
    print("Turns: ", {domain: mean(values) for domain, values in turns.items()})