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

import numpy as np
import torch

from services.service import PublishSubscribe
from services.service import Service
from tools.espnet_minimal.asr.pytorch_backend.asr_init import load_trained_model
from tools.espnet_minimal.nets.batch_beam_search import BatchBeamSearch
from tools.espnet_minimal.nets.beam_search import BeamSearch
from utils.domain.domain import Domain

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SpeechInputDecoder(Service):

    def __init__(self, domain: Domain = "", identifier=None, conversation_log_dir: str = None, use_cuda=False):
        """
        Transforms spoken input from the user to text for further processing.

        Args:
            domain (Domain): Needed for Service, but has no meaning here
            identifier (string): Needed for Service
            conversation_log_dir (string): If this is provided, logfiles will be placed by this Service into the specified directory.
            use_cuda (boolean): Whether or not to run the computations on a GPU
        """
        Service.__init__(self, domain=domain, identifier=identifier)
        self.conversation_log_dir = conversation_log_dir

        # load model
        model_dir = os.path.join(get_root_dir(), "resources", "models", "speech", "multi_en_20190916")
        self.model, conf = load_trained_model(os.path.join(model_dir, "model.bin"))
        self.vocab = conf.char_list

        # setup beam search
        self.bs = BeamSearch(scorers=self.model.scorers(),
                             weights={"decoder": 1.0, "ctc": 0.0},
                             sos=self.model.sos,
                             eos=self.model.eos,
                             beam_size=4,
                             vocab_size=len(self.vocab),
                             pre_beam_score_key="decoder")

        self.bs.__class__ = BatchBeamSearch

        # choose hardware to run on
        if use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.bs.to(self.device)

        # change from training mode to eval mode
        self.model.eval()
        self.bs.eval()

        # scale and offset for feature normalization
        # follows https://github.com/kaldi-asr/kaldi/blob/33255ed224500f55c8387f1e4fa40e08b73ff48a/src/transform/cmvn.cc#L92-L111
        norm = torch.load(os.path.join(model_dir, "cmvn.bin"))
        count = norm[0][-1]
        mean = norm[0][:-1] / count
        var = (norm[1][:-1] / count) - mean * mean
        self.scale = 1.0 / torch.sqrt(var)
        self.offset = - (mean * self.scale)

    @PublishSubscribe(sub_topics=["speech_features"], pub_topics=["gen_user_utterance"])
    def features_to_text(self, speech_features):
        """
        Turns features of the utterance into a string and returns the user utterance in form of text
        
        Args:
            speech_features (np.array): The features that the speech feature extraction module produces
        
        Returns:
            dict(string, string): The user utterance as text
        """
        speech_in_features_normalized = torch.from_numpy(speech_features) * self.scale + self.offset
        with torch.no_grad():
            encoded = self.model.encode(speech_in_features_normalized.to(self.device))
            result = self.bs.forward(encoded)

        # We only consider the most probable hypothesis.
        # Language Model could improve this, right now we don't use one.
        # This might need some post-processing...
        user_utterance = "".join(self.vocab[y] for y in result[0].yseq) \
            .replace("▁", " ") \
            .replace("<space>", " ") \
            .replace("<eos>", "") \
            .strip()

        # write decoded text into logging directory
        if self.conversation_log_dir is not None:
            with open(os.path.join(self.conversation_log_dir, (str(np.math.floor(time.time())) + "_user.txt")),
                      "w") as convo_log:
                convo_log.write(user_utterance)

        print("User: {}\n".format(user_utterance))

        return {'gen_user_utterance': user_utterance}
