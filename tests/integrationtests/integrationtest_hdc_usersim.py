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

"""
Evaluates the performance of the handcrafted policy
"""


import os
import sys
import argparse

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(get_root_dir())

from modules.policy.rl.experience_buffer import UniformBuffer, NaivePrioritizedBuffer

from dialogsystem import DialogSystem
from modules.bst import HandcraftedBST
from modules.simulator import HandcraftedUserSimulator
from utils.domain.jsonlookupdomain import JSONLookupDomain
from modules.policy.evaluation import PolicyEvaluator
from modules.simulator import SimpleNoise
from modules.policy import HandcraftedPolicy
from utils import DiasysLogger, LogLevel
from utils import common


def test_hdc_usersim(domain_name: str, logger: DiasysLogger, eval_epochs: int, eval_dialogs: int,
                     max_turns: int, test_error: float, use_tensorboard: bool):
  
    domain = JSONLookupDomain(domain_name)
    bst = HandcraftedBST(domain=domain, logger=logger)
    user_simulator = HandcraftedUserSimulator(domain, logger=logger)
    noise = SimpleNoise(domain=domain, train_error_rate=0., test_error_rate=test_error, 
                        logger=logger)
    policy= HandcraftedPolicy(domain=domain, logger=logger)
    evaluator = PolicyEvaluator(domain=domain, use_tensorboard=use_tensorboard, 
                                experiment_name='hdc_eval', logger=logger)
    ds = DialogSystem(policy,
                      user_simulator,
                      noise,
                      bst,
                      evaluator
                    )
    ds.eval()
    
    for epoch in range(eval_epochs):
        evaluator.start_epoch()
        for episode in range(eval_dialogs):
            ds.run_dialog(max_length=max_turns)
        evaluator.end_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    domains = ['courses', 'lecturers']
    parser.add_argument("-d", "--domain", required=False, choices=domains,
                    help="The domain which should be used.",
                    default=domains[0])

    parser.add_argument("-lf", "--logtofile", action="store_true", help="log dialog to filesystem")
    parser.add_argument("-lt", "--logtensorboard", action="store_true", 
                        help="log training and evaluation metrics to tensorboard")

    parser.add_argument("-rs", "--randomseed", type=int, default=None)

    parser.add_argument("-e", "--epochs", default=8, type=int,
                        help="number of training and evaluation epochs")
    parser.add_argument("-ed", "--evaldialogs", default=500, type=int,
                        help="number of evaluation dialogs per epoch")
    parser.add_argument("-mt", "--maxturns", default=25, type=int,
                        help="maximum turns per dialog (dialogs with more turns will be terminated and counting as failed")

    parser.add_argument("-eer", "--evalerror", type=float, default=0.0,
                        help="simulation error rate while evaluating")

    args = parser.parse_args()

    # init random generator and logger
    common.init_random(args.randomseed)
    file_log_lvl = LogLevel.DIALOGS if args.logtofile else LogLevel.NONE
    logger = DiasysLogger(console_log_lvl=LogLevel.RESULTS, file_log_lvl=file_log_lvl)

    # choose 'real' domain name from shorthand
    if args.domain == 'courses':
        domain_name = 'ImsCourses'
    elif args.domain == 'lecturers':
        domain_name = 'ImsLecturers'
    
    test_hdc_usersim(domain_name=domain_name, logger=logger, eval_epochs=args.epochs,
                     eval_dialogs=args.evaldialogs, max_turns=args.maxturns,
                     test_error=args.evalerror, use_tensorboard=args.logtensorboard)

    