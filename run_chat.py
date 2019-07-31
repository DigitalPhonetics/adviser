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
This module allows to chat with the dialog system.

NOTE: this is not intended as an integration test!
"""

from utils import common
import os
import argparse
from utils.logger import DiasysLogger, LogLevel

from utils.common import Language
from dialogsystem import DialogSystem
from modules.nlu import HandcraftedNLU
from modules.bst import HandcraftedBST
from modules.nlg import HandcraftedNLG
from modules.policy import DQNPolicy, HandcraftedPolicy
from modules.metapolicy import HandcraftedMetapolicy
from modules.simulator import HandcraftedUserSimulator  # , RNNUserSimulator
from modules.surfaces import ConsoleInput, ConsoleOutput, GuiInput, GuiOutput
from utils.domain.jsonlookupdomain import JSONLookupDomain


def test_domain(domain_name: str, policy_type: str, gui: bool, logger: DiasysLogger,
                language: Language):
    """ Start chat with system.

    Args:
        domain_name (str): name of domain (according to the names in resources/databases)
        policy_type (str): either hdc (handcrafted policy) or dqn (reinforcement learning policy)
        gui (bool): if true, will start a QT GUI session, otherwise the console will be used
                    for interaction
        logger (DiasysLogger): logger for all modules

    .. note::
    
        When using dqn, make sure you have a trained model. You can train a model for the specified
        domain by executing
        
        .. code:: bash
        
            python modules/policy/rl/train_dqnpolicy.py -d domain_name

    """

    # init domain
    domain = JSONLookupDomain(name=domain_name)

    # init modules
    nlu = HandcraftedNLU(domain=domain, logger=logger, language=language)
    bst = HandcraftedBST(domain=domain, logger=logger)

    if policy_type == 'hdc':
        policy = HandcraftedPolicy(domain=domain, logger=logger)
    else:
        policy = DQNPolicy(domain=domain, train_dialogs=1, logger=logger)
        policy.load()

    nlg = HandcraftedNLG(domain=domain, logger=logger, language=language)

    # interaction mode
    if gui:
        input_module = GuiInput(domain, logger=logger)
        output_module = GuiOutput(domain, logger=logger)
    else:
        input_module = ConsoleInput(domain, logger=logger, language=language)
        output_module = ConsoleOutput(domain, logger=logger)

    # construct dialog graph
    ds = DialogSystem(
        input_module,
        nlu,
        bst,
        policy,
        nlg,
        output_module,
        logger=logger)

    # start chat
    ds.eval()
    ds.run_dialog()


def test_multi(logger: DiasysLogger, language: Language):
    domain = JSONLookupDomain(
        'ImsLecturers',
        json_ontology_file=os.path.join('resources', 'databases', 'ImsLecturers-rules.json'),
        sqllite_db_file=os.path.join('resources', 'databases', 'ImsLecturers-dbase.db'))
    l_nlu = HandcraftedNLU(domain=domain, logger=logger, language=language)
    l_bst = HandcraftedBST(domain=domain, logger=logger)
    l_policy = HandcraftedPolicy(domain=domain, logger=logger)
    l_nlg = HandcraftedNLG(domain=domain, logger=logger, language=language)

    lecturers = DialogSystem(
                            l_nlu,
                            l_bst,
                            l_policy,
                            l_nlg,
                            domain=domain,
                            logger=logger
    )
    domain = JSONLookupDomain(
        'ImsCourses',
        json_ontology_file=os.path.join('resources', 'databases', 'ImsCourses-rules.json'),
        sqllite_db_file=os.path.join('resources', 'databases', 'ImsCourses-dbase.db'))
    c_nlu = HandcraftedNLU(domain=domain, logger=logger, language=language)
    c_bst = HandcraftedBST(domain=domain, logger=logger)
    c_policy = HandcraftedPolicy(domain=domain, logger=logger)
    c_nlg = HandcraftedNLG(domain=domain, logger=logger, language=language)

    courses = DialogSystem(
                        c_nlu,
                        c_bst,
                        c_policy,
                        c_nlg,
                        domain=domain,
                        logger=logger
    )

    multi = HandcraftedMetapolicy(
        subgraphs=[courses, lecturers],
        in_module=ConsoleInput(None, logger=logger, language=language),
        out_module=ConsoleOutput(None, logger=logger),
        logger=logger)
    multi.run_dialog()


if __name__ == '__main__':
    domains = ['courses', 'lecturers', 'multi']
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", required=False, choices=domains,
                        help="single- (courses |lecturers ) or multidomain choice",
                        default=domains[0])
    parser.add_argument("-p", "--policy", choices=['hdc', 'dqn'], default="hdc",
                        help="""policy type: handcrafted (hdc) or reinforcement learning (dqn).
                                Note: if using a RL-based policy, make sure you have a trained
                                model (e.g. by executing 
                                    python modules/policy/rl/train_dqnpolicy.py -d your_domain
                                ) """)
    parser.add_argument("-rs", "--randomseed", type=int, default=None)
    parser.add_argument("-g", "--gui", action='store_true',
                        help="show dialogs in the graphical user interface")
    parser.add_argument("-lf", "--logtofile", action="store_true", help="log dialog to filesystem")
    parser.add_argument("-l", "--language", choices=['english', 'german'])
    args = parser.parse_args()

    # init random generator and logger
    common.init_random(args.randomseed)
    file_log_lvl = LogLevel.DIALOGS if args.logtofile else LogLevel.NONE
    logger = DiasysLogger(console_log_lvl=LogLevel.DIALOGS, file_log_lvl=file_log_lvl)

    language = None
    if args.language:
        language = Language[args.language.upper()]

    # choose 'real' domain name from shorthand
    if args.domain == 'multi':
        # multidomain
        test_multi(logger=logger, language=language)
    else:
        if args.domain == 'courses':
            domain_name = 'ImsCourses'
        elif args.domain == 'lecturers':
            domain_name = 'ImsLecturers'

        test_domain(domain_name=domain_name, policy_type=args.policy, gui=args.gui, logger=logger,
                    language=language)
