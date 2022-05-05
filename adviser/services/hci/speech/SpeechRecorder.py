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

import os
import time
import warnings
import wave
from typing import Union

import librosa
import numpy as np
import pyaudio
from matplotlib import pyplot as plt
try:
    from pynput import keyboard  # this needs sudo under MacOS, except if you add python to the accessibility tab
except:
    # make sure that imports work if XServer is not available
    warnings.warn("Could not import pynput, speech recorder will not work.")

from services.service import PublishSubscribe
from services.service import Service
from utils.domain.domain import Domain


class SpeechRecorder(Service):

    def __init__(self, domain: Union[str, Domain] = "", conversation_log_dir: str = None, enable_plotting: bool = False, threshold: int = 8000,
                 voice_privacy: bool = False, identifier: str = None) -> None:
        """
        A service that can record a microphone upon a key pressing event 
        and publish the result as an array. The end of the utterance is 
        detected automatically, also the voice can be masked to alleviate 
        privacy issues.
        
        Args:
            domain (Domain): I don't know why this is here. Service needs it, but it means nothing in this context.
            conversation_log_dir (string): If this parameter is given, log files of the conversation will be created in this directory
            enable_plotting (boolean): If this is set to True, the recorder is no longer real time able and thus the recordings don't work properly. This is just to be used to tune the threshold for the end of utterance detection, not during deployment.
            threshold (int): The threshold below which the assumption of the end of utterance detection is silence
            voice_privacy (boolean): Whether or not to enable the masking of the users voice
            identifier (string): I don't know why this is here. Service needs it.
        """
        Service.__init__(self, domain=domain, identifier=identifier)
        self.conversation_log_dir = conversation_log_dir
        self.recording_indicator = False
        self.audio_interface = pyaudio.PyAudio()
        self.push_to_talk_listener = keyboard.Listener(on_press=self.start_recording)
        self.threshold = threshold
        self.enable_plotting = enable_plotting
        self.voice_privacy = voice_privacy

    @PublishSubscribe(pub_topics=["speech_in"])
    def record_user_utterance(self):
        """
        Records audio once a button is pressed and stops if there is enough continuous silence.
        The numpy array consisting of the frames will be published once it's done.
        
        Returns:
            dict(string, tuple(np.array, int)): The utterance in form of an array and the sampling rate of the utterance
        """
        self.recording_indicator = True
        chunk = 1024  # how many frames per chunk
        audio_format = pyaudio.paInt16  # 16 bit integer based audio for quick processing
        channels = 1  # our asr model only accepts mono sounds
        sampling_rate = 16000  # only 16000 Hz works for the asr model we're using
        stream = self.audio_interface.open(format=audio_format,
                                           channels=channels,
                                           rate=sampling_rate,
                                           input=True,
                                           frames_per_buffer=chunk)
        binary_sequence = []  # this will hold the entire utterance once it's finished as binary data
        # setup for naive end of utterance detection
        continuous_seconds_of_silence_before_utterance_ends = 3.0  # this may be changed freely
        required_silence_length_to_stop_in_chunks = int(
            (continuous_seconds_of_silence_before_utterance_ends * sampling_rate) / chunk)
        reset = int((continuous_seconds_of_silence_before_utterance_ends * sampling_rate) / chunk)
        maximum_utterance_time_in_chunks = int((20 * sampling_rate) / chunk)  # 20 seconds
        if self.enable_plotting:
            threshold_plotter = self.threshold_plotter_generator()
        chunks_recorded = 0
        print("\nrecording...")
        for _ in range(maximum_utterance_time_in_chunks):
            raw_data = stream.read(chunk)
            chunks_recorded += 1
            wave_data = wave.struct.unpack("%dh" % chunk, raw_data)
            binary_sequence.append(raw_data)
            if self.enable_plotting:
                threshold_plotter(wave_data)
            if np.max(wave_data) > self.threshold:
                required_silence_length_to_stop_in_chunks = reset
            else:
                required_silence_length_to_stop_in_chunks -= 1
                if required_silence_length_to_stop_in_chunks == 0:
                    break
        print("...done recording.\n")
        stream.stop_stream()
        stream.close()
        if self.enable_plotting:
            plt.close()
        if self.conversation_log_dir is not None:
            audio_file = wave.open(
                os.path.join(self.conversation_log_dir, (str(np.math.floor(time.time())) + "_user.wav")), 'wb')
            audio_file.setnchannels(channels)
            audio_file.setsampwidth(self.audio_interface.get_sample_size(audio_format))
            audio_file.setframerate(sampling_rate)
            audio_file.writeframes(b''.join(binary_sequence))
            audio_file.close()
        self.recording_indicator = False
        audio_sequence = wave.struct.unpack("%dh" % chunk * chunks_recorded, b''.join(binary_sequence))
        if self.voice_privacy:
            return {"speech_in": (voice_sanitizer(np.array(audio_sequence, dtype=np.float32)), sampling_rate)}
        else:
            return {"speech_in": (np.array(audio_sequence, dtype=np.float32), sampling_rate)}

    def start_recording(self, key):
        """
        This method is a callback of the push to talk key
        listener. It calls the recorder, if it's not already recording.
        
        Args:
            key (Key): The pressed key
        """
        if (key is keyboard.Key.cmd_r or key is keyboard.Key.ctrl_r) and not self.recording_indicator:
            self.record_user_utterance()

    def start_recorder(self):
        """
        Starts the listener and outputs that the speech recorder is ready for use
        """
        self.push_to_talk_listener.start()
        print("To speak to the system, tap your right [CTRL] or [CMD] key.\n"
              "It will try to automatically detect when your utterance is over.\n")

    def threshold_plotter_generator(self):
        """
        Generates a plotter to visualize when the signal is above the set threshold
        
        Returns:
            function: Plots the threshold with the current continuous waveform
        """
        import matplotlib
        matplotlib.use('TkAgg')
        plt.figure(figsize=(10, 2))
        plt.axhline(y=self.threshold, xmin=0.0, xmax=1.0, color='r')
        plt.axhline(y=-self.threshold, xmin=0.0, xmax=1.0, color='r')
        plt.pause(0.000000000001)

        def threshold_plotter(data):
            plt.clf()
            plt.tight_layout()
            plt.axis([0, len(data), -20000, 20000])
            plt.plot(data, color='b')
            plt.axhline(y=self.threshold, xmin=0.0, xmax=1.0, color='r')
            plt.axhline(y=-self.threshold, xmin=0.0, xmax=1.0, color='r')
            plt.pause(0.000000000001)

        return threshold_plotter


def voice_sanitizer(audio):
    """
    While this is by no means a good voice sanitizer,
    it works as a proof of concept. It randomly shifts
    the spectrogram of a speakers utterance up or down,
    making automatic speaker identification much harder
    while keeping impact on asr performance as low as
    possible. The use should be turned off by default.
    
    Args:
        audio (np.array): The audio represented as array
    
    Returns:
        np.array: The mutated audio as array
    """
    spectrogram = librosa.stft(audio)
    voice_shift = np.random.randint(3, 6)
    if np.random.choice([True, False]):
        for frequency_index, _ in enumerate(spectrogram):
            # mutate the voice to be higher
            try:
                spectrogram[len(spectrogram) - (frequency_index + 1)] = spectrogram[
                    len(spectrogram) - (frequency_index + 1 + voice_shift)]
            except IndexError:
                pass
    else:
        for frequency_index, _ in enumerate(spectrogram):
            # mutate the voice to be lower
            try:
                spectrogram[frequency_index] = spectrogram[frequency_index + voice_shift]
            except IndexError:
                pass

    return librosa.istft(spectrogram)
