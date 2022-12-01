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
This module allows to chat with the dialog system.
"""

import argparse
import os
from services.service import ControlChannelMessages

from services.bst import HandcraftedBST
from services.domain_tracker.domain_tracker import DomainTracker
from utils.logger import DiasysLogger, LogLevel
from services.dialogsystem import DialogSystem


def load_console():
    from services.hci.console import ConsoleInput, ConsoleOutput
    user_in = ConsoleInput(domain="")
    user_out = ConsoleOutput(domain="")
    return [user_in, user_out]


def load_nlg(backchannel: bool, domain = None):
    if backchannel:
        from services.nlg import BackchannelHandcraftedNLG
        nlg = BackchannelHandcraftedNLG(domain=domain, sub_topic_domains={'predicted_BC': ''})
    else:
        from services.nlg.nlg import HandcraftedNLG
        nlg = HandcraftedNLG(domain=domain)
    return nlg


def load_mensa_domain(backchannel: bool = False):
    from examples.webapi.mensa import MensaDomain, MensaNLU
    from services.policy.policy_api import HandcraftedPolicy as PolicyAPI
    mensa = MensaDomain()
    mensa_nlu = MensaNLU(domain=mensa)
    mensa_bst = HandcraftedBST(domain=mensa)
    mensa_policy = PolicyAPI(domain=mensa)
    mensa_nlg = load_nlg(backchannel=backchannel, domain=mensa)
    return mensa, [mensa_nlu, mensa_bst, mensa_policy, mensa_nlg]


def load_lecturers_domain(backchannel: bool = False):
    from utils.domain.jsonlookupdomain import JSONLookupDomain
    from services.nlu.nlu import HandcraftedNLU
    from services.nlg.nlg import HandcraftedNLG
    from services.policy import HandcraftedPolicy
    domain = JSONLookupDomain('ImsLecturers', display_name="Lecturers")
    lect_nlu = HandcraftedNLU(domain=domain)
    lect_bst = HandcraftedBST(domain=domain)
    lect_policy = HandcraftedPolicy(domain=domain)
    lect_nlg = load_nlg(backchannel=backchannel, domain=domain)
    return domain, [lect_nlu, lect_bst, lect_policy, lect_nlg]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ADVISER 2.0 Dialog System')
    parser.add_argument('domains', nargs='+', choices=['lecturers', 'weather', 'mensa', 'qa'],
                        help="Chat domain(s). For multidomain type as list: domain1 domain2 .. \n",
                        default="ImsLecturers")
    parser.add_argument('-g', '--gui', action='store_true', help="Start Webui server")
    parser.add_argument('--asr', action='store_true', help="enable speech input")
    parser.add_argument('--tts', action='store_true', help="enable speech output")
    parser.add_argument('--bc', action='store_true', help="enable backchanneling (doesn't work with 'weather' domain")
    parser.add_argument('--debug', action='store_true', help="enable debug mode")
    parser.add_argument('--log_file', choices=['dialogs', 'results', 'info', 'errors', 'none'], default="none",
                        help="specify file log level")
    parser.add_argument('--log', choices=['dialogs', 'results', 'info', 'errors', 'none'], default="results",
                        help="specify console log level")
    parser.add_argument('--cuda', action='store_true', help="enable cuda (currently only for asr/tts)")
    parser.add_argument('--privacy', action='store_true',
                        help="enable random mutations of the recorded voice to mask speaker identity", default=False)
    # TODO option for remote services
    # TODO option for video
    # TODO option for multiple consecutive dialogs 
    args = parser.parse_args()
    if args.bc and not args.asr:
        parser.error("--bc: Backchannel requires ASR (--asr) option")

    num_dialogs = 100
    domains = []
    services = []

    # setup logger
    file_log_lvl = LogLevel[args.log_file.upper()]
    log_lvl = LogLevel[args.log.upper()]
    conversation_log_dir = './conversation_logs'
    speech_log_dir = None
    if file_log_lvl == LogLevel.DIALOGS:
        # log user audio, system audio and complete conversation
        import time
        from math import floor

        print("This Adviser call will log all your interactions to files.\n")
        if not os.path.exists(f"./{conversation_log_dir}"):
            os.mkdir(f"./{conversation_log_dir}/")
        conversation_log_dir = "./" + conversation_log_dir + "/{}/".format(floor(time.time()))
        os.mkdir(conversation_log_dir)
        speech_log_dir = conversation_log_dir
    logger = DiasysLogger(file_log_lvl=file_log_lvl,
                          console_log_lvl=log_lvl,
                          logfile_folder=conversation_log_dir,
                          logfile_basename="full_log")

    # load domain specific services
    if 'lecturers' in args.domains:
        l_domain, l_services = load_lecturers_domain(backchannel=args.bc)
        domains.append(l_domain)
        services.extend(l_services)
    if 'mensa' in args.domains:
        m_domain, m_services = load_mensa_domain(backchannel=args.bc)
        domains.append(m_domain)
        services.extend(m_services)

   
    services.extend(load_console())


    # setup dialog system
    services.append(DomainTracker(domains=domains))
    debug_logger = logger if args.debug else None

    ds = DialogSystem(services=services)
    ds.run(start_messages={"user_utterance.ImsLecturers":""})
    # ds.run(start_messages={ControlChannelMessages.DIALOG_END: False})
   