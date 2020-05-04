###############################################################################
#
# Copyright 2020, University of Stuttgart:
# Institute for Natural Language Processing (IMS)
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

"""Feature extraction with openSMILE.

This module provides a feature extractor which uses the openSMILE toolkit
to extract features from raw audio. The user utterance which is represented
as a numpy array in-memory needs to be written to a temporary file first, so
that openSMILE can read it.

"""

import librosa
import numpy as np
import os
import subprocess

from services.service import PublishSubscribe
from services.service import Service
from tools.getopensmile import get_opensmile_executable_path
from scipy.io.wavfile import write


class SpeechFeatureExtractor(Service):
    """SpeechFeatureExtractor calls openSMILE to extract features from audio.

    Note: openSMILE will be downloaded & compiled to tools/opensmile
    if not found there.
    """

    def __init__(self):
        """ SpeechFeatureExtractor.

        The following things are setup on initialization:
        * directory for temporary audio files
        * path to openSMILE config files
        * path to openSMILE executable
        """
        Service.__init__(self)
        self.speech_out_dir = os.path.join("resources",
                                           "tmp_audio_and_features")
        self.cfg_dir = os.path.join("resources", "opensmile_config")
        self.openSmile_path = get_opensmile_executable_path()

    @PublishSubscribe(sub_topics=['speech_in'],
                      pub_topics=['gemaps_features', 'mfcc_features'])
    def speech_to_features(self, speech_in):
        """Wrapper function for feature extraction.

        This function writes the user utterance to an audio file and calls
        extract_wav_file_features for each feature config.

        Args:
            speech_in (numpy.ndarray, int): Tuple containing user utterance
                and sampling rate

        Returns:
            dict: extracted features, keys must match pub_topics
        """
        # sampling_rate depends on SpeechRecorder
        tmp_file = os.path.join(self.speech_out_dir, 'tmp_audio.wav')
        write(tmp_file, speech_in[1], speech_in[0].astype(np.int16))
        # TODO(delete tmp_file after all opensmile features are extracted)
        feature_configs = {'gemaps_features': 'gemaps/GeMAPSv01a',
                           'mfcc_features': 'features_for_BC/mfcc'}
        # Extracting acoustic features using openSMILE
        features = {}
        for feature_type in feature_configs:
            config = feature_configs[feature_type]
            features[feature_type] = self.extract_wav_file_features(
                config,
                tmp_file
            )

        return features

    def remove_file(self, file_name):
        """ Remove specified file.

        Args:
            file_name (str): full path of file which shall be removed
        """
        try:
            os.remove(file_name)
        except FileNotFoundError as error:
            self.logger.error(error)
            raise (error)

    def extract_wav_file_features(self, features, new_audio_file):
        """Extracting acoustic features using openSMILE.

        Args:
            features (str): path to openSMILE's feature config
            new_audio_file (str): path to audio file

        Returns:
            numpy.ndarray: extracted features for the audio file
        """
        output_file = new_audio_file + '.csv'
        config_file = os.path.join(self.cfg_dir, features + ".conf")
        f = open(os.devnull, 'w')
        try:
            # OpenSMILE command to extract features
            # SMILExtract -C <configfile> -I <input_file> −O <output_file>
            command = ' '.join([self.openSmile_path,
                                '-C', config_file,
                                '-I', new_audio_file,
                                '-csvoutput', output_file,
                                '-headercsv', '0',
                                '-timestampcsv', '0',
                                '-instname', '0'])
            subprocess.call(command, stdout=f, stderr=f, shell=True)
            return self.preprocess_csv(output_file)

        except OSError as err:
            print(command)
            print("OS error: {0}".format(err))

    def preprocess_csv(self, csv_file):
        """Get features from csv file and normalize them if necessary.

        openSMILE feature are written to temporary csv file. This function
        reads them into a numpy array, removes instance names and could
        do a normalization step if needed. This is not implemented right now.

        Args:
            csv_file (str): path to csv file

        Returns:
            numpy.ndarray: raw feature values
        """
        feats = np.genfromtxt(csv_file, delimiter=';')
        if len(feats.shape) == 1:
            # reshape one-dimensional features, e.g. gemaps
            feats = feats.reshape(1, -1)
        # take everything except first column which is the instance name
        feats = feats[:, 1:]
        # StandardScaler normalizes feats to zero mean and unit variance
        # for frame-wise LLDs, it will standardize across frames
        # for gemaps (or functionals in general), it's not possible to scale
        # ideal setup: fit StandardScaler on training set across samples
        # and apply it with .transform() to new samples
        # scaler = preprocessing.StandardScaler()
        self.remove_file(csv_file)
        return feats
