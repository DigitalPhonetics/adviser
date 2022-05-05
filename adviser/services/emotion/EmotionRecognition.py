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

"""Emotion recognition module."""

import warnings
import numpy as np
import os
import pickle
import torch
from torch.nn.functional import softmax
import torchaudio
from torchaudio.compliance.kaldi import fbank
from services.service import PublishSubscribe
from services.service import Service
from utils.userstate import EmotionType
try:
    from resources.models.emotion.emotion_cnn import cnn
except:
    warnings.warn("Could not import additional resources. Some parts might not work until you run `sh download_models.sh`.")


class EmotionRecognition(Service):
    """Emotion recognition module.

    This module receives acoustic features, loads pretrained models and outputs
    predictions of emotional states. It can easily be extended/adapted to use
    different models and facial features in addition.
    """

    def __init__(self):
        """ Emotion recognition module.

        On initialization all necessary models are loaded.
        """
        Service.__init__(self)
        self.emotion_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.abspath(
            os.path.join(
                self.emotion_dir, "..", "..", "resources", "models", "emotion"
            )
        )

        def load_args(emo_representation):
            arg_dict = pickle.load(
                open(os.path.join(
                    self.model_path, f'{emo_representation}_args.pkl'),
                     'rb')
            )
            return arg_dict

        def load_model(emo_representation, arg_dict):
            ARGS = arg_dict['args']
            model = cnn(
                kernel_size=(ARGS.height, arg_dict['D_in']),
                D_out=arg_dict['D_out'],
                args=ARGS
            )
            model.load_state_dict(
                torch.load(
                    os.path.join(self.model_path,
                                 f'{emo_representation}_model_params.pt'),
                    map_location=torch.device('cpu')
                )
            )
            model.eval()
            return model

        self.emo_representations = ['category', 'arousal', 'valence']
        self.models = {}
        self.args = {}
        for emo_representation in self.emo_representations:
            self.args[emo_representation] = load_args(emo_representation)
            self.models[emo_representation] = load_model(
                emo_representation,
                self.args[emo_representation]
            )
        self.arousal_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        self.valence_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.category_mapping = {
            0: EmotionType.Angry,
            1: EmotionType.Happy,
            2: EmotionType.Neutral,
            3: EmotionType.Sad
        }

    @PublishSubscribe(sub_topics=["fbank"], pub_topics=["emotion"])
    def predict_from_audio(self, fbank):
        """Emotion prediction from acoustic features.

        Args:
            fbank (torch.Tensor): feature array, shape (sequence, num_mel_bins)

        Returns:
            dict: nested dictionary containing all results, main key: 'emotion'
        """

        def normalize_and_pad_features(features: torch.Tensor, seq_len, mean: torch.Tensor, std: torch.Tensor):
            # normalize
            features = (features - mean) / std
            # cut or pad with zeros as necessary
            features = torch.cat(
                [features[:seq_len],  # take feature data until :seq_len
                 features.new_zeros(  # pad with zeros if seq_len > feature.size(0)
                    (seq_len - features.size(0)) if seq_len > features.size(0) else 0,
                     features.size(1))],
                dim=0  # concatenate zeros in time dimension
            )
            return features

        predictions = {}
        for emo_representation in self.emo_representations:
            seq_len = self.args[emo_representation]['args'].seq_length
            mean = self.args[emo_representation]['norm_mean']
            std = self.args[emo_representation]['norm_std']
            # feature normalization and padding has to be done for each
            # emotion representation individually because the means and
            # standard (deviations) (and sequence length) can be different
            features = normalize_and_pad_features(fbank, seq_len, torch.from_numpy(mean), torch.from_numpy(std))
            predictions[emo_representation] = softmax(
                self.models[emo_representation](features.unsqueeze(1)), dim=1
            ).detach().numpy()

        arousal_level = self.arousal_mapping[np.argmax(predictions['arousal'])]
        valence_level = self.valence_mapping[np.argmax(predictions['valence'])]
        category_label = self.category_mapping[np.argmax(predictions['category'])]

        return {'emotion': {'arousal': arousal_level,
                            'valence': valence_level,
                            'category': category_label,
                            'cateogry_probabilities':
                                np.around(predictions['category'], 2).reshape(-1)}}


if __name__ == '__main__':
    recognition = EmotionRecognition()
    waveform, sample_rate = torchaudio.load_wav('test.wav')
    f = fbank(waveform, sample_frequency=sample_rate)
    emo = recognition.predict_from_audio(f)
    print(emo)
