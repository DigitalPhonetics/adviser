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
from argparse import Namespace

import nltk
import torch
import yaml
from g2p_en import G2p
from parallel_wavegan.models import ParallelWaveGANGenerator
from typing import Dict

from services.hci.speech.cleaners import custom_english_cleaners
from services.service import PublishSubscribe
from services.service import Service
from tools.espnet_minimal.asr.asr_utils import get_model_conf
from tools.espnet_minimal.asr.asr_utils import torch_load
from tools.espnet_minimal.utils import dynamic_import
from utils.domain.domain import Domain


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    
class SpeechOutputGenerator(Service):
    
    def __init__(self, domain: Domain = "", identifier: str = None, use_cuda=False, sub_topic_domains: Dict[str, str] = {}):
        """
        Text To Speech Module that reads out the system utterance.
        
        Args:
            domain (Domain): Needed for Service, no meaning here
            identifier (string): Needed for Service
            use_cuda (boolean): Whether or not to perform computations on GPU. Highly recommended if available
            sub_topic_domains: see `services.service.Service` constructor for more details
        """
        Service.__init__(self, domain=domain, identifier=identifier, sub_topic_domains=sub_topic_domains)
        self.models_directory = os.path.join(get_root_dir(), "resources", "models", "speech")

        # The following lines can be changed to incorporate different models.
        # This is the only thing that needs to be changed for that, everything else should be dynamic.
        self.transcription_type = "phn"
        self.dict_path = os.path.join(self.models_directory,
                                      "phn_train_no_dev_pytorch_train_fastspeech.v4", "data", "lang_1phn",
                                      "train_no_dev_units.txt")
        self.model_path = os.path.join(self.models_directory,
                                       "phn_train_no_dev_pytorch_train_fastspeech.v4", "exp",
                                       "phn_train_no_dev_pytorch_train_fastspeech.v4", "results",
                                       "model.last1.avg.best")
        self.vocoder_path = os.path.join(self.models_directory,
                                         "ljspeech.parallel_wavegan.v1", "checkpoint-400000steps.pkl")
        self.vocoder_conf = os.path.join(self.models_directory, "ljspeech.parallel_wavegan.v1", "config.yml")

        # define device to run the synthesis on
        if use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # define end to end TTS model
        self.input_dimensions, self.output_dimensions, self.train_args = get_model_conf(self.model_path)
        model_class = dynamic_import.dynamic_import(self.train_args.model_module)
        model = model_class(self.input_dimensions, self.output_dimensions, self.train_args)
        torch_load(self.model_path, model)
        self.model = model.eval().to(self.device)
        self.inference_args = Namespace(**{"threshold": 0.5, "minlenratio": 0.0, "maxlenratio": 10.0})

        # define neural vocoder
        with open(self.vocoder_conf) as vocoder_config_file:
            self.config = yaml.load(vocoder_config_file, Loader=yaml.Loader)
        vocoder = ParallelWaveGANGenerator(**self.config["generator_params"])
        vocoder.load_state_dict(torch.load(self.vocoder_path, map_location="cpu")["model"]["generator"])
        vocoder.remove_weight_norm()
        self.vocoder = vocoder.eval().to(self.device)

        with open(self.dict_path) as dictionary_file:
            lines = dictionary_file.readlines()
        lines = [line.replace("\n", "").split(" ") for line in lines]
        self.char_to_id = {c: int(i) for c, i in lines}
        self.g2p = G2p()

        # download the pretrained Punkt tokenizer from NLTK. This is done only
        # the first time the code is executed on a machine, if it has been done
        # before, this line will be skipped and output a warning. We will probably
        # redirect warnings into a file rather than std_err in the future, since
        # there's also a lot of pytorch warnings going on etc.
        nltk.download('punkt', quiet=True)

    def preprocess_text_input(self, text):
        """
        Clean the text and then convert it to id sequence.
        
        Args:
            text (string): The text to preprocess
        """
        text = custom_english_cleaners(text)  # cleans the text
        if self.transcription_type == "phn":  # depending on the model type, different preprocessing is needed.
            text = filter(lambda s: s != " ", self.g2p(text))
            text = " ".join(text)
            char_sequence = text.split(" ")
        else:
            char_sequence = list(text)
        id_sequence = []
        for c in char_sequence:
            if c.isspace():
                id_sequence += [self.char_to_id["<space>"]]
            elif c not in self.char_to_id.keys():
                id_sequence += [self.char_to_id["<unk>"]]
            else:
                id_sequence += [self.char_to_id[c]]
        id_sequence += [self.input_dimensions - 1]  # <eos>
        return torch.LongTensor(id_sequence).view(-1).to(self.device)

    @PublishSubscribe(sub_topics=["sys_utterance"], pub_topics=["system_speech"])
    def generate_speech(self, sys_utterance):
        """
        Takes the system utterance and turns it into a sound
        
        Args:
            sys_utterance (string): The new system utterance
        
        Returns:
            dict(string, tuple(np.array, int, string)): Everything needed to play the system utterance as an audio and the utterance in text for logging
        """
        with torch.no_grad():
            preprocessed_text_as_list = self.preprocess_text_input(sys_utterance)
            features_from_text, _, _ = self.model.inference(preprocessed_text_as_list, self.inference_args)
            feature_dimension = features_from_text.size(0) * self.config["hop_size"]
            random_tensor_with_proper_dimensions = torch.randn(1, 1, feature_dimension).to(self.device)
            auxiliary_content_window = self.config["generator_params"]["aux_context_window"]
            preprocessed_features = features_from_text.unsqueeze(0).transpose(2, 1)
            features_from_text = torch.nn.ReplicationPad1d(auxiliary_content_window)(preprocessed_features)
            generated_speech = self.vocoder(random_tensor_with_proper_dimensions, features_from_text).view(-1)
            sound_as_array = generated_speech.view(-1).cpu().numpy()
        return {"system_speech": (sound_as_array, self.config["sampling_rate"], sys_utterance)}
