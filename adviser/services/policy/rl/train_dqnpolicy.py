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

"""
# This script can be executed to train a DQN policy.
# It will create a policy model (file ending with .pt).

# You need to execute this script before you can interact with the RL agent.
# """

import os
import sys
import argparse

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


sys.path.append(get_root_dir())

from services.policy.rl.experience_buffer import NaivePrioritizedBuffer, UniformBuffer

from services.bst import HandcraftedBST
from services.simulator import HandcraftedUserSimulator
from services.policy import DQNPolicy
from services.stats.evaluation import PolicyEvaluator
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils import DiasysLogger, LogLevel
from services.service import DialogSystem
from torch.utils.tensorboard import SummaryWriter

from utils import common



def train(domain_name: str, log_to_file: bool, seed: int, train_epochs: int, train_dialogs: int,
          eval_dialogs: int, max_turns: int, train_error_rate: float, test_error_rate: float,
          lr: float, eps_start: float, grad_clipping: float, buffer_classname: str,
          buffer_size: int, use_tensorboard: bool):

    """
        Training loop for the RL policy, for information on the parameters, look at the descriptions
        of commandline arguments in the "if main" below
    """
    seed = seed if seed != -1 else None
    common.init_random(seed=seed)

    file_log_lvl = LogLevel.DIALOGS if log_to_file else LogLevel.NONE
    logger = DiasysLogger(console_log_lvl=LogLevel.RESULTS, file_log_lvl=file_log_lvl)

    summary_writer = SummaryWriter(log_dir='logs') if use_tensorboard else None
    
    if buffer_classname == "prioritized":
        buffer_cls = NaivePrioritizedBuffer
    elif buffer_classname == "uniform":
        buffer_cls = UniformBuffer

    domain = JSONLookupDomain(name=domain_name)
    
    bst = HandcraftedBST(domain=domain, logger=logger)
    user = HandcraftedUserSimulator(domain, logger=logger)
    # noise = SimpleNoise(domain=domain, train_error_rate=train_error_rate,
    #                     test_error_rate=test_error_rate, logger=logger)
    policy = DQNPolicy(domain=domain, lr=lr, eps_start=eps_start,
                    gradient_clipping=grad_clipping, buffer_cls=buffer_cls,
                    replay_buffer_size=buffer_size, train_dialogs=train_dialogs,
                    logger=logger, summary_writer=summary_writer)
    evaluator = PolicyEvaluator(domain=domain, use_tensorboard=use_tensorboard,
                                experiment_name=domain_name, logger=logger,
                                summary_writer=summary_writer)
    ds = DialogSystem(services=[user, bst, policy, evaluator], protocol='tcp')
    # ds.draw_system_graph()

    error_free = ds.is_error_free_messaging_pipeline()
    if not error_free:
        ds.print_inconsistencies()

    for j in range(train_epochs):
        # START TRAIN EPOCH
        evaluator.train()
        policy.train()
        evaluator.start_epoch()
        for episode in range(train_dialogs):
            if episode % 100 == 0:
                print("DIALOG", episode)
            logger.dialog_turn("\n\n!!!!!!!!!!!!!!!! NEW DIALOG !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            ds.run_dialog(start_signals={f'user_acts/{domain.get_domain_name()}': []})
        evaluator.end_epoch()
        policy.save()

        # START EVAL EPOCH
        evaluator.eval()
        policy.eval()
        evaluator.start_epoch()
        for episode in range(eval_dialogs):
            logger.dialog_turn("\n\n!!!!!!!!!!!!!!!! NEW DIALOG !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            ds.run_dialog(start_signals={f'user_acts/{domain.get_domain_name()}': []})
        evaluator.end_epoch()
    ds.shutdown()


if __name__ == "__main__":
    # possible test domains
    domains = ['lecturers']

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", required=False, choices=domains,
                        help="lecturers",
                        default=domains[0])
    parser.add_argument("-lf", "--logtofile", action="store_true", help="log dialog to filesystem")
    parser.add_argument("-lt", "--logtensorboard", action="store_true",
                        help="log training and evaluation metrics to tensorboard")

    parser.add_argument("-rs", "--randomseed", default=12345, type=int,
                        help="seed for random generator initialization; use -1 for a randomly generated seed")

    parser.add_argument("-e", "--epochs", default=8, type=int,
                        help="number of training and evaluation epochs")
    parser.add_argument("-td", "--traindialogs", default=1000, type=int,
                        help="number of training dialogs per epoch")
    parser.add_argument("-ed", "--evaldialogs", default=500, type=int,
                        help="number of evaluation dialogs per epoch")
    parser.add_argument("-mt", "--maxturns", default=25, type=int,
                        help="maximum turns per dialog (dialogs with more turns will be terminated and counting as failed")

    parser.add_argument("-ter", "--trainerror", type=float, default=0.0,
                        help="simulation error rate while training")
    parser.add_argument("-eer", "--evalerror", type=float, default=0.0,
                        help="simulation error rate while evaluating")

    parser.add_argument("-lr", "--learningrate", type=float, default=0.0001,
                        help="learning rate for optimization algorithm")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.7,
                        help="initial exploration rate (epsilon) - in [0,1]")
    parser.add_argument("-cg", "--clipgrad", type=float, default=0.0,
                        help="upper bound gradient is going to be clipped to")

    parser.add_argument("-bn", "--buffername", choices=['uniform', 'prioritized'],
                        help="experience replay buffer type", default='prioritized')
    parser.add_argument("-bs", "--buffersize", type=int, default=8192,
                        help="capacity of experience replay buffer")
    args = parser.parse_args()
    assert 0 <= args.epsilon <= 1, "exploration rate has to be between 0 and 1"

    if args.domain == 'lecturers':
        domain_name = 'ImsLecturers'

    train(domain_name=domain_name, log_to_file=args.logtofile,
          use_tensorboard=args.logtensorboard, seed=args.randomseed, train_epochs=args.epochs,
          train_dialogs=args.traindialogs, eval_dialogs=args.evaldialogs, max_turns=args.maxturns,
          train_error_rate=args.trainerror, test_error_rate=args.evalerror, lr=args.learningrate,
          eps_start=args.epsilon, grad_clipping=args.clipgrad, buffer_classname=args.buffername,
          buffer_size=args.buffersize
          )
