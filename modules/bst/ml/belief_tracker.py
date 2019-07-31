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

# this code is based on glorannas master thesis
import os
import string
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

import  modules.bst.ml.config_helper as ch
import  modules.bst.ml.dstc_data as sd
from modules.module import Module
from utils.sysact import SysAct
from utils.useract import UserAct
from utils.beliefstate import BeliefState



class InformableTracker(nn.Module):
    def __init__(self, vocab, slot_name, slot_value_count,
                 glove_embedding_dim=300, gru_dim=100, dense_output_dim=50,
                 p_dropout=0.5):
        super(InformableTracker, self).__init__()
        self.weight_file_name = os.path.join(ch.get_weight_path(), 'belieftracker_' + slot_name + '.weights')

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.dense_output_dim = dense_output_dim
        self.gru_dim = gru_dim
        self.glove_embedding_dim = glove_embedding_dim

        self.embedding_dim = glove_embedding_dim + self.vocab_size + 2
        
        # word-level gru
        self.gru = nn.GRU(self.embedding_dim, gru_dim)
        self.gru_dropout = nn.Dropout()

        # output layer(s)
        modules = []
        current_dim = gru_dim * 2
        if dense_output_dim > 0:
            modules.append(nn.Linear(current_dim, dense_output_dim))
            modules.append(nn.ReLU())
            current_dim = dense_output_dim
        modules.append(nn.Linear(current_dim, slot_value_count[slot_name]))
        self.output_layers = nn.ModuleList(modules)

        print(self)

    def _init_hidden(self):
        return torch.zeros(1, 1, self.gru_dim, 
                       dtype=torch.float, device=ch.DEVICE, requires_grad=True)

    def _embed_sentence(self, utterance, user=False):
        # return sentece_len x embedding_dim
        sentence_emb = []

        tagged_utterance = [sd.TOKEN_BOT] + utterance.split() + [sd.TOKEN_EOT]
        for word in tagged_utterance:
            actor_emb = torch.tensor([float(user==True), float(user==True)],
                                    dtype=torch.float, device=ch.DEVICE)
            keyword_emb = torch.zeros(self.vocab_size, dtype=torch.float,
                                      device=ch.DEVICE)
            if word in self.vocab:
                keyword_emb[self.vocab[word]] = 1.0
            else:
                keyword_emb[self.vocab[sd.TOKEN_UNK]] = 1.0
            glove_emb = sd.glove_embed(word, self.glove_embedding_dim)
            sentence_emb.append(torch.cat([actor_emb, keyword_emb, glove_emb],
                                dim=0).unsqueeze(dim=0))
        
        return torch.cat(sentence_emb, dim=0)

    def forward(self, sys_sentence, usr_sentence, first_turn=False):
        if first_turn:
            self.h_t = self._init_hidden()
        
        # process system turn
        sys_input = self._embed_sentence(sys_sentence, user=False).unsqueeze(1)
        _, self.h_t = self.gru(sys_input, self.h_t)
        h_sys = self.gru_dropout(self.h_t).squeeze(0)

        # process user turn
        usr_input = self._embed_sentence(usr_sentence, user=True).unsqueeze(1)
        _, self.h_t = self.gru(usr_input, self.h_t)
        h_usr = self.gru_dropout(self.h_t).squeeze(0)

        h_concat = torch.cat([h_sys, h_usr], dim=1) # 1 x 2*hidden
        output = h_concat

        # project into output space
        for layer in self.output_layers:
            output = layer(output)
        
        return output


    def save(self):
        #print("saving state tracker weights for slot " + self.slot_name + "...")
        torch.save(self.state_dict(), self.weight_file_name)

    def load(self, model_path=""):
        #print("loading state tracker weights for slot " + self.slot_name + "...")
        file = os.path.join(model_path, self.weight_file_name)
        self.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))




class RequestableTracker(nn.Module):
    def __init__(self, vocab, requestable_slot,
                 glove_embedding_dim=300, gru_dim=100, dense_output_dim=50,
                 p_dropout=0.5):
        super(RequestableTracker, self).__init__()
        self.weight_file_name = os.path.join(ch.get_weight_path(), 'belieftracker_req_' + requestable_slot + '.weights')

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.dense_output_dim = dense_output_dim
        self.gru_dim = gru_dim
        self.glove_embedding_dim = glove_embedding_dim
        self.slot_name = requestable_slot
     
        self.embedding_dim = glove_embedding_dim + self.vocab_size + 2
        
        # word-level gru
        self.gru = nn.GRU(self.embedding_dim, gru_dim)
        self.gru_dropout = nn.Dropout()

        # output layer(s)
        modules = []
        current_dim = gru_dim * 2
        if dense_output_dim > 0:
            modules.append(nn.Linear(current_dim, dense_output_dim))
            modules.append(nn.ReLU())
            current_dim = dense_output_dim
        self.output_layers = nn.ModuleList(modules)
        self.output_binary = nn.Linear(current_dim, 2)

        print(self)

    def _init_hidden(self):
        return torch.zeros(1, 1, self.gru_dim, 
                       dtype=torch.float, device=ch.DEVICE, requires_grad=True)

    def _embed_sentence(self, utterance, user=False):
        # return sentece_len x embedding_dim
        sentence_emb = []

        tagged_utterance = [sd.TOKEN_BOT] + utterance.split() + [sd.TOKEN_EOT]
        for word in tagged_utterance:
            actor_emb = torch.tensor([float(user==True), float(user==True)],
                                    dtype=torch.float, device=ch.DEVICE)
            keyword_emb = torch.zeros(self.vocab_size, dtype=torch.float,
                                      device=ch.DEVICE)
            if word in self.vocab:
                keyword_emb[self.vocab[word]] = 1.0
            else:
                keyword_emb[self.vocab[sd.TOKEN_UNK]] = 1.0
            glove_emb = sd.glove_embed(word, self.glove_embedding_dim)
            sentence_emb.append(torch.cat([actor_emb, keyword_emb, glove_emb],
                                dim=0).unsqueeze(dim=0))
        
        return torch.cat(sentence_emb, dim=0)

    def forward(self, sys_sentence, usr_sentence, first_turn=False):
        if first_turn:
            self.h_t = self._init_hidden()
        
        # process system turn
        sys_input = self._embed_sentence(sys_sentence, user=False).unsqueeze(1)
        _, self.h_t = self.gru(sys_input, self.h_t)
        h_sys = self.gru_dropout(self.h_t).squeeze(0)

        # process user turn
        usr_input = self._embed_sentence(usr_sentence, user=True).unsqueeze(1)
        _, self.h_t = self.gru(usr_input, self.h_t)
        h_usr = self.gru_dropout(self.h_t).squeeze(0)

        h_concat = torch.cat([h_sys, h_usr], dim=1) # 1 x 2*hidden
        output = h_concat

        # project into output space
        for layer in self.output_layers:
            output = layer(output)
        # binary prediciton for requestable slot
        output = self.output_binary(output)
        return output


    def save(self):
        #print("saving state tracker weights for slot " + self.slot_name + "...")
        torch.save(self.state_dict(), self.weight_file_name)

    def load(self, model_path=""):
        #print("loading state tracker weights for slot " + self.slot_name + "...")
        file = os.path.join(model_path, self.weight_file_name)
        self.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))

