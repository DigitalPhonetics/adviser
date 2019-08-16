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


import os
import json
import sys
import pickle
import string
import re
import jsonpickle
import copy
import random
import spacy

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(get_root_dir())

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn
from utils.domain.jsonlookupdomain import JSONLookupDomain

try:
    nlp = spacy.load('en_core_web_sm')
except:
    pass

import modules.bst.ml.config_helper as ch

# directory of the DTSC dataset (here: inside /data)
DOMAIN = JSONLookupDomain('CamRestaurants')
DATA_DIR = ch.get_data_path()
rules_json = os.path.join(DATA_DIR, 'dstc2', 'traindev', 'scripts', 'config', 'ontology_dstc2.json')

TOKEN_BOT = '<BOT>' # begin of turn token
TOKEN_EOT = '<EOT>' # end of turn token
TOKEN_UNK = '<UNK>' # unnknown word


def delexicalise_utterance(utterance, informable_slotvalues, requestable_slots):
    delexed = copy.deepcopy(utterance)
    for inf_slot in informable_slotvalues:
        for inf_val in informable_slotvalues[inf_slot]:
            if inf_val.lower() != '**none**':
                # replace values
                if inf_slot == 'addr':
                    inf_slot = 'address'
                pattern = re.compile("(?<!\#(v.|s.))" + inf_val.lower())
                delexed = pattern.sub('#v.' + inf_slot.lower() + '#', delexed)
    for inf_slot in informable_slotvalues:
        # replace slots
        if inf_slot == 'addr':
            inf_slot = 'address'
        pattern = re.compile("(?<!\#(v.|s.))" + inf_slot.lower())
        delexed = pattern.sub('#s.' + inf_slot.lower() + '#', delexed)
    for req_slot in requestable_slots:
        # replace slots
        if req_slot == 'addr':
            req_slot = 'address'
        pattern = re.compile("(?<!\#(v.|s.))" + req_slot.lower())
        delexed = pattern.sub('#s.' + req_slot.lower() + '#', delexed)
    return delexed


class User(object):
    """ User's part of a dialog turn """
    def __init__(self, user_goal, requested_slots, method, utterance, 
                 informable_slotvalues, requestable_slots):
        # sometimes pricerange is spelled price range -> convert to same format
        if "price range" in user_goal:
            user_goal["pricerange"] = user_goal["price range"]
            user_goal.pop("price range")

        self.user_goal = user_goal
        transcript = nlp(utterance.lower()) # tokenize
        self.utterance = " ".join([tok.text for tok in transcript])

        # delexicalise user utterance for intent tracker
        self.delexicalised_utterance = delexicalise_utterance(self.utterance, 
                                      informable_slotvalues, requestable_slots)
        

        self.requested_slots = requested_slots
        # sometimes pricerange is spelled price range -> convert to same format
        for i, req_slot in enumerate(self.requested_slots):
            if req_slot == "price range":
                self.requested_slots[i] = "pricerange"

        self.method = method

class DialogAct(object):
    def __init__(self, act, slots):
        self.act = act
        self.slots = slots
    
class Slot(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class System(object):
    """ System's part of a dialog turn """
    def __init__(self, system_json):
        if 'dialog-acts' in system_json:
            self.acts = []
            for act in system_json['dialog-acts']:
                slots = []
                for slot in act['slots']:
                    # coerce the value to a string and lowercase it
                    slot[1] = str(slot[1]).lower()

                    slots.append(Slot(slot[0], slot[1]))
                self.acts.append(DialogAct(act['act'], slots))
            self.input_txt = sysact_to_utterance_triple(self.acts, True, False)
        else:
            self.input_txt = ""
            self.acts = []

class DialogTurn(object):
    """ System and user parts of a dialog turn """
    def __init__(self, turn, labels, informable_slotvalues, requestable_slots):
        self.index = turn['turn-index']
        # parse system turn
        self.system = System(turn['output'])
        # parse user turn
        self.user = User(
            labels['goal-labels'],
            labels['requested-slots'],
            labels['method-label'],
            labels.get('transcription', ""),
            informable_slotvalues, requestable_slots
        )

class Dialog(object):
    def __init__(self, log, labels, informable_slotvalues, requestable_slots):
        self.turns = [] 
        for turn_log, turn_label in zip(log['turns'], labels['turns']):
            self.turns.append(DialogTurn(turn_log, turn_label, 
                                     informable_slotvalues, requestable_slots))
    
    def __len__(self):
        return len(self.turns)




def _init_only_once():
    global glove_embeddings 
    glove_embeddings = {}

# embedding_dim has to be 50, 100, 200 or 300
def _load_glove_embedding(embedding_dim, data_path=""):
    _init_only_once()
    assert(embedding_dim == 50 or embedding_dim == 100 or embedding_dim == 200 or embedding_dim == 300)
    if not embedding_dim in glove_embeddings:
        # load glove embedding
        pretrained_glove_emb_file = os.path.join(data_path, ch.get_data_path(), 'glove.6B', 'glove.6B.' + str(embedding_dim) + 'd.txt')
        print("reading glove embedding from file " + pretrained_glove_emb_file + " ... ")
        #glove_embeddings[embedding_dim] = pd.read_table(pretrained_glove_emb_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        embedding = {}
        with open(pretrained_glove_emb_file,'r') as f:
            for line in f:
                # format: 1.st entry is word, rest is tensor
                line = line.split()
                embedding[line[0]] = torch.tensor([float(value) for value in line[1:]], dtype=torch.float, device=ch.DEVICE)
        glove_embeddings[embedding_dim] = embedding 
        print("done")
    else:
        print("glove embedding of size " + str(embedding_dim) + " is already loaded")

def glove_embed(word, embedding_dim):
    if not 'glove_embeddings' in globals():
        # init global embedding storage
        _init_only_once()
    if not embedding_dim in glove_embeddings:
        _load_glove_embedding(embedding_dim)
     # glove embedding
    input_word_ascii = word.encode('ascii') # panda uses ascii for index keys
    if input_word_ascii in glove_embeddings[embedding_dim]:
        return glove_embeddings[embedding_dim][input_word_ascii]
    else:
        return torch.zeros(embedding_dim, dtype=torch.float, device=ch.DEVICE, requires_grad=True)


def sysact_to_utterance_triple(acts, replaceActs, removeUnusedActs):
    utterance = []
    for act in acts:
        # modify acts according to rules
        if replaceActs:
            utterance.extend(replaceSystemActs(act, removeUnusedActs))

        # do not alter the acts
        elif len(act.slots) > 0:
            for slot_name, slot_value in act.slots:
                utterance.append(act.act)
                utterance.append(slot_name)
                utterance.append(slot_value)
        else:
            utterance.append("%s" % act.act)

    if len(utterance) == 0:
        return ""
    else:
        if replaceActs:
            # remove duplicate acts
            utterance = sorted(set(utterance), key=utterance.index)
        return " ".join(utterance)


def replaceSystemActs(actObj, removeUnusedActs):
    """ Replace weird system act names to be recognizable by word embeddings """
    keepSlotValues = True
    act = actObj.act.lower()

    if act == "expl-conf":
        act = "confirm"
    elif act == "impl-conf":
        act = "assume"

    if removeUnusedActs:
        if act in ["bye", "canthear", "confirm-domain", "repeat", "reqmore", "inform", "offer", "welcomemsg", "select"] or act.startswith("canthelp"):
            act = ""
            keepSlotValues = False
    elif act == "welcomemsg":
        act = "welcome"
    elif act == "canthear":
        act = "silence"
    elif act == "confirm-domain":
        act = "really restaurant ?"
    elif act == "reqmore":
        act = "need more ?"
    elif act.startswith("canthelp"): # canthelp, canthelp.exception
        act = "unavailable"
        keepSlotValues = False
    # name is ok but remove slots and values
    elif act in ["inform", "offer", "select"]: # select is informative, but it only appears in the dev set, not in train, test -> remove
        keepSlotValues = False

    #print "act after processing", act

    if act == "":
        return []

    result = [act]
    if keepSlotValues:
        for i, slot in enumerate(actObj.slots):
            if i > 0:
                result.append(act)
            result.append(slot.name)
            result.append(slot.value)
    return result



class DSTC2Data(object):
    def __init__(self, path_to_data_folder="", preprocess=False, load_train_data=True):  
        """
        Args:
            preprocess: if True, preprocess first, else try loading existing 
                        preprocessed data
            load_train_data: if True, the training data set will be loaded - 
                        otherwise only vocabularys
        """
        
        self.vocabulary = {}
        self.inf_slots = set()
        self.req_slots = {}
        self.method_slots = set() 
        self.inf_values = {} # dict of value -> index mapping for each slot (key)
    
        self.train_data = []
        self.dev_data = []
        self.test_data = []
        
        self.data_train_file = os.path.join(path_to_data_folder, ch.get_preprocessing_path(), DOMAIN.get_domain_name() +  '_train.json')
        self.data_dev_file = os.path.join(path_to_data_folder, ch.get_preprocessing_path(), DOMAIN.get_domain_name() +  '_dev.json')
        self.data_test_file = os.path.join(path_to_data_folder, ch.get_preprocessing_path(), DOMAIN.get_domain_name() +  '_test.json')
        self.slotval_file = os.path.join(path_to_data_folder, ch.get_preprocessing_path(), DOMAIN.get_domain_name() + '_slotvals.json')
        self.vocabulary_file = os.path.join(path_to_data_folder, ch.get_preprocessing_path(), DOMAIN.get_domain_name() + '_vocab.json')
        
        if preprocess:
            self._extract_dialogs_from_directory()
        else:
            self._load_preprocessed_data(load_train_data)
    
    def get_train_data(self):
        return self.train_data

    def get_train_len(self):
        return len(self.train_data)
    
    def get_dev_data(self):
        return self.dev_data

    def get_dev_len(self):
        return len(self.dev_data)

    def get_test_data(self):
        return self.test_data

    def get_test_len(self):
        return len(self.test_data)

    def get_vocabulary(self):
        return self.vocabulary
    
    def count_informable_slot_values(self, slot):
        return len(self.inf_values[slot])

    def get_informable_slot_value_index(self, slot, value):
        return self.inf_values[slot][value]

    def get_informable_slot_value(self, slot, index):
        for value in self.inf_values[slot]:
            if self.inf_values[slot][value] == index:
                return value
        return None

    def get_requestable_slots(self):
        return list(self.req_slots.keys())

    def get_requestable_slot_index(self, slot):
        return self.req_slots[slot]
    
    def get_requestable_slot(self, index):
        for slot in self.req_slots:
            if self.req_slots[slot] == index:
                return slot
        return None

    def get_method_slots(self):
        return self.method_slots

    def get_method_index(self, method):
        return list(self.method_slots).index(method)

    def get_method_name(self, index):
        return list(self.method_slots)[index]

    def _get_random_slot_value(self, slot, current_value):
        if current_value == '**NONE**' or current_value == 'dontcare':
            return current_value
        rand_val = random.choice(list(self.inf_values[slot].keys()))
        while rand_val == current_value or rand_val == '**NONE**' or rand_val == 'dontcare':
            rand_val = random.choice(list(self.inf_values[slot].keys()))
        return rand_val

    def _extract_dialogs_from_directory(self):
        print("starting preprocessing of restaurant dialogs...")

        self.vocabulary[TOKEN_UNK] = len(self.vocabulary)
        self.vocabulary[TOKEN_BOT] = len(self.vocabulary)
        self.vocabulary[TOKEN_EOT] = len(self.vocabulary)

        dialog_count = 0
        turn_count = 0
       
        # parse rules file
        with open(rules_json) as json_rules_file:
            rules = json.load(json_rules_file)
            for requestable in rules['requestable']:
                if requestable == 'addr':
                    requestable = 'address'
                if not requestable in self.vocabulary:
                    self.vocabulary[requestable] = len(self.vocabulary)
                    self.vocabulary["#s." + requestable + "#"] = len(self.vocabulary)
                if not requestable in self.req_slots:
                    self.req_slots[requestable] = len(self.req_slots)
            for informable in rules['informable']:
                if informable == 'addr':
                    informable = 'address'
                if not informable in self.vocabulary:
                    self.vocabulary[informable] = len(self.vocabulary)
                    self.vocabulary["#s." + informable + "#"] = len(self.vocabulary)
                self.inf_slots.add(informable)
                self.inf_values[informable] = {'**NONE**': 0, 'dontcare': 1}
                for slot_value in rules['informable'][informable]:
                    if not slot_value in self.vocabulary:
                        self.vocabulary[slot_value] = len(self.vocabulary)
                        self.vocabulary["#v." + slot_value + "#"] = len(self.vocabulary)
                    self.inf_values[informable][slot_value] = len(self.inf_values[informable])
                
            for method in rules['method']:
                self.method_slots.add(method)

       
        # parse data files
        traindev_dir = os.path.join(DATA_DIR, 'dstc2', 'traindev')
        test_dir = os.path.join(DATA_DIR, 'dstc2', 'test')
        train_files = os.path.join(traindev_dir, 'scripts', 'config', 'dstc2_train.flist')
        dev_files = os.path.join(traindev_dir, 'scripts', 'config', 'dstc2_dev.flist')
        test_files = os.path.join(test_dir, 'scripts', 'config', 'dstc2_test.flist')
        with open(train_files) as f:
            files = [line.rstrip('\n') for line in f]
            for filename in files:
                # get the two files belonging to the same dialog
                label_file_name = os.path.join(traindev_dir, 'data', filename, 'label.json')
                log_file_name = os.path.join(traindev_dir, 'data', filename, 'log.json')
                turn_count += self._extract_dialog_from_json(label_file_name, log_file_name, self.train_data)
                dialog_count += 1
        with open(dev_files) as f:
            files = [line.rstrip('\n') for line in f]
            for filename in files:
                # get the two files belonging to the same dialog
                label_file_name = os.path.join(traindev_dir, 'data', filename, 'label.json')
                log_file_name = os.path.join(traindev_dir, 'data', filename, 'log.json')
                turn_count += self._extract_dialog_from_json(label_file_name, log_file_name, self.dev_data)
                dialog_count += 1
        with open(test_files) as f:
            files = [line.rstrip('\n') for line in f]
            for filename in files:
                # get the two files belonging to the same dialog
                label_file_name = os.path.join(test_dir, 'data', filename, 'label.json')
                log_file_name = os.path.join(test_dir, 'data', filename, 'log.json')
                turn_count += self._extract_dialog_from_json(label_file_name, log_file_name, self.test_data)
                dialog_count += 1

        print("train data: processed " + str(dialog_count) + " dialogues with " + str(turn_count) + " turns")


        jsonpickle.set_encoder_options('json', sort_keys=False, indent=4)
        print("writing preprocessed data to file " + self.data_train_file + " ... ")
        with open(self.data_train_file, 'w') as f:
            f.write(jsonpickle.encode(self.train_data))
        print("writing preprocessed data to file " + self.data_dev_file + " ... ")
        with open(self.data_dev_file, 'w') as f:
            f.write(jsonpickle.encode(self.dev_data))
        print("writing preprocessed data to file " + self.data_test_file + " ... ")
        with open(self.data_test_file, 'w') as f:
            f.write(jsonpickle.encode(self.test_data))
        print("writing slots to file " + self.slotval_file + " ...")
        with open(self.slotval_file, 'w') as f:
            f.write(jsonpickle.encode([self.inf_slots, self.inf_values, self.req_slots, self.method_slots]))
        print("writing vocabulary to file " + self.vocabulary_file + " ...")
        with open(self.vocabulary_file, 'w') as f:
            f.write(jsonpickle.encode(self.vocabulary)) 
        print("done")

    def _extract_dialog_from_json(self, label_file_name, log_file_name, dataset):
        turn_count = 0

        dialog = None
        with open(label_file_name) as label_data_file, open(log_file_name) as log_data_file:
            label_data = json.load(label_data_file)
            log_data = json.load(log_data_file)
            dialog = Dialog(log_data, label_data, self.inf_values, self.req_slots)
            dataset.append(dialog)
            return len(dialog.turns)
        return 0

    def _load_preprocessed_data(self, load_train_data):
    
        if load_train_data:
            print("loading preprocessed data from file " + self.data_train_file + " ...")
            with open(self.data_train_file, 'r') as f:
                self.train_data = jsonpickle.decode(f.read())
            print("  train dialogs: ", len(self.train_data))
            print("loading preprocessed data from file " + self.data_dev_file + " ...")
            with open(self.data_dev_file, 'r') as f:
                self.dev_data = jsonpickle.decode(f.read())
            print("  dev dialogs: ", len(self.dev_data))
            print("loading preprocessed data from file " + self.data_test_file + " ...")
            with open(self.data_test_file, 'r') as f:
                self.test_data = jsonpickle.decode(f.read())
            print("  test dialogs: ", len(self.test_data))
            
        print("loading preprocessed slots/values from file " + self.slotval_file + " ...")
        with open(self.slotval_file, 'r') as f:
            slotvallist = jsonpickle.decode(f.read())
            self.inf_slots = slotvallist[0]
            self.inf_values = slotvallist[1]
            self.req_slots = slotvallist[2]
            self.method_slots = slotvallist[3]
            print("loaded slots:")
            print(" - informables", self.inf_slots)
            print(" - requestables", self.req_slots)
            print(" - methods", self.method_slots)
       
        print("loading vocabulary from file " + self.vocabulary_file + " ...")
        with open(self.vocabulary_file, 'r') as f:
            self.vocabulary = jsonpickle.decode(f.read())

        print("done")



if __name__ == "__main__":
    # preprocessing
    data = DSTC2Data(preprocess=True, load_train_data=True)
    print("done")

