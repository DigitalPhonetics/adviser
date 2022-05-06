###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
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

from typing import List
import os
import json
import torch
# from bert_embedding import BertEmbedding
import numpy as np

from utils.logger import DiasysLogger
from utils.domain.lookupdomain import LookupDomain
from utils import UserAct, UserActionType
from utils.sysact import SysAct
from utils.beliefstate import BeliefState
from utils.common import Language
from services.service import PublishSubscribe, Service

from .neuralmodels.simpledot import SimpleDot
from .neuralmodels.tagger import Tagger, extract_entities
from .neuralmodels.director import Classifier

from transformers import BertTokenizerFast, BertModel

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class QuestionParser(Service):
    """Semantic parsing module for question answering

    Attributes:
        device (torch.device): PyTorch device object, either CPU or GPU
        nn_relation (nn.Module): neural network for relation prediction
        nn_entity (nn.Module): neural network for topic entity prediction
        nn_direction (nn.Module): neural network for relation direction prediction
        tags (List[str]): relation tags
        max_seq_len (int): maximum number of tokens per question
        embedding_creator (BertEmbedding): object creating BERT embeddings
    """

    def __init__(self, domain: LookupDomain, \
        logger: DiasysLogger = DiasysLogger(), device: str = 'cpu', cache_dir: str = None):
        """Creates neural networks for semantic parsing and other required utils

        Args:
            domain: the QA domain
            logger: the logger
            device: PyTorch device name
            cache_dir: the cache directory for transformers' models
        """
        Service.__init__(self, domain=domain, debug_logger=logger)
        self.device = torch.device(device)
        self.nn_relation = self._load_relation_model()
        self.nn_entity = self._load_entity_model()
        self.nn_direction = self._load_direction_model()

        self.tags = self._load_tag_set()

        self.max_seq_len = 40
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        self.embedder = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)

    def _load_relation_model(self):
        model = SimpleDot(100, 400, True).to(self.device)
        model.load_state_dict(torch.load(os.path.join(get_root_dir(), 'resources', 'models', 'qa', 'simpledot.pt'), map_location=self.device))
        model.eval()
        return model

    def _load_entity_model(self):
        model = Tagger(100, 400)
        model.load_state_dict(torch.load(os.path.join(get_root_dir(), 'resources', 'models', 'qa', 'tagger.pt'), map_location=self.device))
        model.eval()
        return model

    def _load_direction_model(self):
        model = Classifier(75, 500, 2).to(self.device)
        model.load_state_dict(torch.load(os.path.join(get_root_dir(), 'resources', 'models', 'qa', 'director.pt'), map_location=self.device))
        model.eval()
        return model

    def _load_tag_set(self):
        csqa_tags = []
        with open(os.path.join(get_root_dir(), 'resources', 'ontologies', 'qa', 'csqa_tags.json')) as f:
            csqa_tags = json.load(f)
        return csqa_tags

    @PublishSubscribe(sub_topics=["user_utterance"], pub_topics=["user_acts"])
    def parse_user_utterance(self, user_utterance: str = None) -> dict(user_acts=List[UserAct]):
        """Parses the user utterance.

        Responsible for detecting user acts with their respective slot-values from the user
        utterance by predicting relation, topic entities and the relation's direction.

        Args:
            user_utterance: the last user input as string

        Returns:
            A dictionary with the key "user_acts" and the value containing a list of user actions
        """
        result = {}
        self.user_acts = []

        user_utterance = user_utterance.strip()

        if not user_utterance:
            return {'user_acts': None}
        elif user_utterance.lower().replace(' ', '').endswith('bye'):
            return {'user_acts': [UserAct(user_utterance, UserActionType.Bye)]}
        
        if self.domain.get_keyword() in user_utterance.lower():
            self.user_acts.append(UserAct(user_utterance, UserActionType.SelectDomain))
            begin_idx = user_utterance.lower().find(self.domain.get_keyword())
            user_utterance = user_utterance.lower().replace(self.domain.get_keyword(), "")
            if len(user_utterance) == 0:
                return {'user_acts': self.user_acts}

        tokens, embeddings = self._preprocess_utterance(user_utterance)

        relation_out = self._predict_relation(embeddings)
        entities_out = self._predict_topic_entities(embeddings)
        direction_out = self._predict_direction(embeddings)

        relation_pred = self._lookup_relation(relation_out)
        entities_pred = extract_entities(tokens, entities_out[:,0])
        direction_pred = self._lookup_direction(direction_out)

        self.user_acts.extend([
            UserAct(user_utterance, UserActionType.Inform, 'relation', relation_pred, 1.0),
            UserAct(user_utterance, UserActionType.Inform, 'direction', direction_pred, 1.0)
        ])
        for t in entities_pred:
            self.user_acts.append(UserAct(user_utterance, UserActionType.Inform, 'topic', self.tokenizer.convert_tokens_to_string(t), 1.0))

        result['user_acts'] = self.user_acts
        self.debug_logger.dialog_turn("User Actions: %s" % str(self.user_acts))
        return result

    def _preprocess_utterance(self, utterance):
        encoded_input = self.tokenizer(utterance, return_tensors='pt', truncation=True, max_length=self.max_seq_len)
        with torch.no_grad():
            embeddings: torch.Tensor = self.embedder(**encoded_input).last_hidden_state
        embeddings = torch.cat((embeddings, embeddings.new_zeros(1,self.max_seq_len - embeddings.size(1),768)), dim=1)
        return self.tokenizer.tokenize(utterance, add_special_tokens=False), embeddings.permute(1,0,2)

    def _predict_relation(self, embeddings):
        with torch.no_grad():
            rel_scores = self.nn_relation(embeddings)
        _, pred_rel = torch.max(rel_scores, 1)
        return pred_rel

    def _lookup_relation(self, prediction):
        return self.tags[int(prediction[0])]

    def _predict_topic_entities(self, embeddings):
        with torch.no_grad():
            ent_scores = self.nn_entity(embeddings)
        _, preds_ent = torch.max(ent_scores, 2)
        return preds_ent

    def _predict_direction(self, embeddings):
        with torch.no_grad():
            tag_scores = self.nn_direction(embeddings)
        _, preds = torch.max(tag_scores, 1)
        return preds

    def _lookup_direction(self, prediction):
        return ['out', 'in'][int(prediction[0])]
