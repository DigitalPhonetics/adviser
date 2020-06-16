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
This module runs the demo with speech
"""

import os
import subprocess

from services.service import RemoteService
from examples.webapi.mensa import MensaDomain, MensaNLU
from examples.webapi.weather import WeatherNLU, WeatherNLG, WeatherDomain
from services.policy.policy_api import HandcraftedPolicy as PolicyAPI
from services.bst import HandcraftedBST
from services.domain_tracker.domain_tracker import DomainTracker
from services.hci.console import ConsoleInput, ConsoleOutput
from services.hci.speech import SpeechInputDecoder, SpeechInputFeatureExtractor, SpeechOutputGenerator
from services.hci.speech import SpeechOutputPlayer, SpeechRecorder
from services.hci.video.VideoInput import VideoInput
from services.engagement.engagement_tracker import EngagementTracker
from services.emotion.EmotionRecognition import EmotionRecognition
from services.ust.ust import HandcraftedUST
from services.nlg import HandcraftedNLG
from services.backchannel import AcousticBackchanneller
from services.nlg import BackchannelHandcraftedNLG
from services.nlg import HandcraftedEmotionNLG
from services.nlu import HandcraftedNLU
from services.policy import HandcraftedPolicy
from services.policy.affective_policy import EmotionPolicy
from services.service import DialogSystem
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger, LogLevel
from services.simulator.emotion_simulator import EmotionSimulator
from utils.userstate import EmotionType


# load domains
lecturers = JSONLookupDomain(name='ImsLecturers', display_name="Lecturers")
weather = WeatherDomain()
mensa = MensaDomain()

# only debug logging
conversation_log_dir = "./conversation_logs"
os.makedirs(f"./{conversation_log_dir}/", exist_ok=True)
logger = DiasysLogger(file_log_lvl=LogLevel.NONE,
                        console_log_lvl=LogLevel.DIALOGS,
                        logfile_basename="full_log")

# input modules
user_in = ConsoleInput(conversation_log_dir=conversation_log_dir)
user_out = ConsoleOutput()
recorder = SpeechRecorder(conversation_log_dir=conversation_log_dir)
speech_in_decoder = SpeechInputDecoder(conversation_log_dir=conversation_log_dir, use_cuda=False)#RemoteService(identifier="asr")

# feature extraction
d_tracker = DomainTracker(domains=[lecturers, weather, mensa], greet_on_first_turn=True)
speech_in_feature_extractor = SpeechInputFeatureExtractor()

# feature processing
engagement = EngagementTracker()
emotion = EmotionRecognition()
ust = HandcraftedUST()
affective_policy = EmotionPolicy()

# lecturer specific modules
lect_nlu = HandcraftedNLU(domain=lecturers)
# lect_nlg = HandcraftedEmotionNLG(domain=lecturers, sub_topic_domains={'sys_emotion': '', 'sys_engagement': ''},
#                                  emotions=["Happy", "Angry", "Sad"])
lect_nlg = BackchannelHandcraftedNLG(domain=lecturers, sub_topic_domains={'predicted_BC': ''})

lect_policy = HandcraftedPolicy(domain=lecturers)
lect_bst = HandcraftedBST(domain=lecturers)

# weather specific modules
weather_nlu = WeatherNLU(domain=weather)
weather_nlg = WeatherNLG(domain=weather)
weather_bst = HandcraftedBST(domain=weather)
weather_policy = PolicyAPI(domain=weather)

backchanneler = AcousticBackchanneller()


# install node modules if missing
webui_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "webui"))
if not os.path.isdir(os.path.join(webui_folder, 'node_modules')):
    print("INFO: Couldn't find node dependencies - trying to install...")
    subprocess.run(["npm", "install"], cwd=webui_folder, capture_output=True)
if not os.path.isdir(os.path.join(webui_folder, 'node_modules')):
    print(
        "ERROR: Could not install node dependencies. Make sure node and npm are installed on your machine and you have rights to install node modules via npm.")
    exit()
subprocess.Popen(["npm", "run", "start",  "--silent"], cwd=webui_folder) # NOTE: if you see no messages in the browser app, comment out this line 
                                                                         # and run `npm run start` from `tools/webui` before executing this script
guiserver_proc = subprocess.Popen(["python", "services/hci/gui.py"], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
webui = RemoteService(identifier="GUIServer")

# mensa specific modules
mensa_nlu = MensaNLU(domain=mensa)
mensa_nlg = HandcraftedEmotionNLG(domain=mensa, sub_topic_domains={'sys_emotion': '', 'sys_engagement': ''},
                                  emotions=["Happy", "Angry", "Sad"])
mensa_bst = HandcraftedBST(domain=mensa)
mensa_policy = PolicyAPI(domain=mensa)

# output modules  #
speech_out_generator = SpeechOutputGenerator(domain="", use_cuda=False)  # (GPU: 0.4 s/per utterance, CPU: 11 s/per utterance)
speech_out_player = SpeechOutputPlayer(domain="", conversation_log_dir=conversation_log_dir)

# start the recording listener
recorder.start_recorder()

# create a dialogsystem from the modules
ds = DialogSystem(
    services=[
              user_in,
              recorder,
              backchanneler,
              webui,
              speech_in_feature_extractor,
              speech_in_decoder,
              engagement,
              emotion,
              d_tracker,
              ust,
              affective_policy,
              lect_bst,
              lect_nlg,
              lect_nlu,
              lect_policy,
              weather_bst,
              weather_nlg,
              weather_nlu,
              weather_policy,
              mensa_bst,
              mensa_nlg,
              mensa_nlu,
              mensa_policy,
              user_out,
              speech_out_generator,
              speech_out_player
              ])#, debug_logger=logger)

# view meta information about the system
error_free = ds.is_error_free_messaging_pipeline()
if not error_free:
    ds.print_inconsistencies()
ds.draw_system_graph()

# start dialog
for _ in range(1):
    ds.run_dialog({'gen_user_utterance': ""})
ds.shutdown()
