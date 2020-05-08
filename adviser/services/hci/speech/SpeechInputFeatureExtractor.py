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


import torch
from typing import Tuple

from services.service import PublishSubscribe
from services.service import Service
from utils.domain.domain import Domain
import torchaudio
import numpy

class SpeechInputFeatureExtractor(Service):

    def __init__(self, domain: Domain = ""):
        """
        Given a sound, this service extracts features and passes them on to the decoder for ASR

        Args:
            domain (Domain): Needed for Service, no meaning here
        """
        Service.__init__(self, domain=domain)

    @PublishSubscribe(sub_topics=["speech_in"], pub_topics=["speech_features"])
    def speech_to_features(self, speech_in: Tuple[numpy.array, int]):
        """
        Turns numpy array with utterance into features

        Args:
            speech_in (tuple(np.array), int): The utterance, represented as array and the sampling rate

        Returns:
            np.array: The extracted features of the utterance
        """
        sample_frequence = speech_in[1]
        speech_in = torch.from_numpy(speech_in[0]).unsqueeze(0)

        filter_bank = torchaudio.compliance.kaldi.fbank(speech_in, num_mel_bins=80, sample_frequency=sample_frequence)
        # Default ASR model uses 16kHz, but different models are possible, then the sampling rate only needs to be changd in the recorder
        pitch = torch.zeros(filter_bank.shape[0], 3)  # TODO: check if torchaudio pitch function is better
        speech_in_features = torch.cat([filter_bank, pitch], 1).numpy()

        return {'speech_features': speech_in_features}

    @PublishSubscribe(sub_topics=["speech_in"], pub_topics=["mfcc"])
    def speech_to_mfcc(self, speech_in):
        """
        Extracts 13 Mel Frequency Cepstral Coefficients (MFCC) from input utterance.

        Args:
            speech_in (tuple(np.array), int): The utterance, represented as array and the sampling rate

        Returns:
            np.array: The extracted features of the utterance
        """
        speech = torch.from_numpy(speech_in[0]).unsqueeze(0)
        mfcc = torchaudio.compliance.kaldi.mfcc(
            speech,
            sample_frequency=speech_in[1]
        )
        return {'mfcc': mfcc}

    @PublishSubscribe(sub_topics=["speech_in"], pub_topics=["fbank"])
    def speech_to_fbank(self, speech_in):
        """
        Extracts 23 filterbanks from input utterance.

        Args:
            speech_in (tuple(np.array), int): The utterance, represented as array and the sampling rate

        Returns:
            np.array: The extracted features of the utterance
        """
        speech = torch.from_numpy(speech_in[0]).unsqueeze(0)
        fbank = torchaudio.compliance.kaldi.fbank(
            speech,
            sample_frequency=speech_in[1]
        )
        return {'fbank': fbank}
