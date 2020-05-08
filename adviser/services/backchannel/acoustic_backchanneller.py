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

import numpy as np
import torch
import os
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")
gpu = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

from services.service import PublishSubscribe
from services.service import Service
from services.backchannel.PytorchAcousticBackchanneler import PytorchAcousticBackchanneler


class AcousticBackchanneller(Service):
    """AcousticBackchanneller predicts a backchannel given the last user utterance.
       The model can predict: No backchannel (0), Assessment (1), Continuer (2)
       The backchannel realization is added in the NLG module.
    """

    def __init__(self):
        Service.__init__(self)
        self.speech_in_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        self.trained_model_path = os.path.join('resources', 'models', 'backchannel') + '/pytorch_acoustic_backchanneller.pt'
        self.load_model()

    def load_model(self):
        """
        The PyTorch Backchannel model is instantiated and the pretrained parameters are loaded.

        Returns:
        """
        self.model = PytorchAcousticBackchanneler()
        self.model.load_state_dict(torch.load(self.trained_model_path))
        self.model.eval()

    def split_input_data(self, mfcc_features):
        """
        Preprocess and segmentation of MFCC features of the user's speech.
        Segmentation is done every 150ms without overlapping.

        Args:
            mfcc_features (numpy.array): mffcc features of users speech

        Returns:
            new_data (list): segmented mfcc features

        """
        input_height = 150  # this stands for 150ms
        input_length = mfcc_features.shape[0]
        zero_shape = list(mfcc_features.shape)
        zero_shape[0] = input_height
        ranges = list(reversed([idx for idx in range(input_length - 1, 0, -input_height)]))
        new_data = []
        for r in ranges:
            if r < input_height:
                zero_data = np.zeros(zero_shape)
                zero_data[-r:, :] = mfcc_features[:r, :]
                new_data.append(zero_data)
            else:
                new_data.append(mfcc_features[r - input_height:r, :])
        return (new_data)

    @PublishSubscribe(sub_topics=['mfcc'],
                      pub_topics=["predicted_BC"])
    def backchannel_prediction(self, mfcc: np.array):
        """
        Service that receives the MFCC features from the user's speech.
        It preprocess and normalize them and makes the BC prediction.

        Args:
            mfcc_features (torch.tensor): MFCC features

        Returns:
            (dict): a dictionary with the key "predicted_BC" and the value of the BC type
        """
        # class_int_mapping = {0: b'no_bc', 1: b'assessment', 2: b'continuer'}
        mfcc_features = mfcc.numpy()
        scaler = preprocessing.StandardScaler()
        mfcc_features = scaler.fit_transform(mfcc_features)
        input_splits = self.split_input_data(mfcc_features)
        prediction = self.model(input_splits).detach().numpy().argmax(axis=1)

        # Returning the majority, unless a BC appears,
        if len(set(prediction)) == 1:
            return {'predicted_BC':  prediction[0]}
        elif 1 in prediction and 2 in prediction:
            ones = len(prediction[prediction==1])
            twos = len(prediction[prediction==2])
            return {'predicted_BC':  1 if ones > twos else 2}
        else:
            return {'predicted_BC': 1 if 1 in prediction else 2}

