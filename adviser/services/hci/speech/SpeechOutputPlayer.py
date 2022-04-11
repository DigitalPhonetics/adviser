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


import math
import os
import time

import librosa
import sounddevice

from services.service import PublishSubscribe
from services.service import Service
from utils.domain.domain import Domain
import soundfile as sf


class SpeechOutputPlayer(Service):

    def __init__(self, domain: Domain = "", conversation_log_dir: str = None, identifier: str = None):
        """
        Service that plays the system utterance as sound
        
        Args:
            domain (Domain): Needed for Service, but has no meaning here
            conversation_log_dir (string): If this is provided it will create log files in the specified directory.
            identifier (string): Needed for Service.
        """
        Service.__init__(self, domain=domain, identifier=identifier)
        self.conversation_log_dir = conversation_log_dir
        self.interaction_count = 0

    @PublishSubscribe(sub_topics=["system_speech"], pub_topics=[])
    def speak(self, system_speech):
        """
        Takes the system utterance and reads it out. Also can log the audio and text.
        
        Args:
            system_speech (np.array): An array of audio that is meant to produce a sound from. The result of the systems TTS synthesis service.
        """
        sounddevice.play(system_speech[0], system_speech[1])

        # log the utterance
        if self.conversation_log_dir is not None:
            file_path = os.path.join(self.conversation_log_dir, (str(math.floor(time.time()))))
            sf.write(file_path + "_system.wav", system_speech[0], system_speech[1], 'PCM_24')
            with open(file_path + "_system.txt", "w") as convo_log:
                convo_log.write(system_speech[2])
