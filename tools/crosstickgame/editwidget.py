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

import sys
import os
import copy
import json

from PyQt5 import uic
from PyQt5.QtWidgets import QWidget

from utils import UserAct, UserActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain
from tools.crosstickgame.nlumodules import NLU_MODULES


VALUABLE_USER_ACTS = [
    frozenset([UserAct(act_type=UserActionType.Hello, score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Inform, slot='obligatory_attendance', value='false', score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Inform, slot='name', value='information retrieval and text mining', score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Inform, slot='master', value='true', score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Inform, slot='module_name', value='computational linguistics team laboratory', score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Request, slot='lecturer', score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Request, slot='id', score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Request, slot='time_slot', score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Request, slot='ects', score=1.0)]),
    frozenset([UserAct(act_type=UserActionType.Bye, score=1.0)])
]


class EditWidget(QWidget):
    """The widget for the MinMax game.

    Inside the edit widget, the user can select a set of user acts and 
    
    Attributes:
        _domain {Domain} -- the domain of the NLU module
        _module_name {str} -- the name of the NLU module
        _module {Module} -- the NLU module
        _file_out {str} -- the file name to write the changes to
        _curr_user_acts {frozenset[UserAct]} -- the current set of user acts as specified in the UserActView
        _test_cases {dict[frozenset[UserAct], list[str]]} -- a list of test cases for each user act set
    """
    def __init__(self, save_to, domain, module_name, parent = None):
        QWidget.__init__(self, parent)
        self._ui = uic.loadUi(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui/editwidget.ui'), self)

        self._domain = domain
        self._module_name = module_name
        self._module = NLU_MODULES[domain][module_name]

        self._file_out = save_to
        
        self._ui.user_acts.set_domain(domain)
        self._curr_user_acts = frozenset(self._ui.user_acts.get_user_acts())
        self._ui.test_cases.set_module(self._module)
        self._ui.test_cases.set_sentences([], self._curr_user_acts)

        self._test_cases = {}
        
        self._ui.user_acts.user_acts_changed.connect(self._on_user_acts_changed)
        self._ui.test_cases.tests_changed.connect(self._on_test_cases_changed)
    
    def _analyse_sentence(self, user_acts, sentence):
        result = self._module.forward(None, user_utterance=sentence)['user_acts']
        analysed_user_acts = frozenset(result)
        return analysed_user_acts == user_acts
    
    def _count_points(self):
        points = 0
        for user_act in VALUABLE_USER_ACTS:
            if user_act in self._test_cases:
                counter = 0
                for sentence in self._test_cases[user_act]:
                    if sentence.strip() == '':
                        continue
                    if not self._analyse_sentence(user_act, sentence):
                        counter += 1
                        if counter == 1:
                            points += 10
                        elif counter == 2:
                            points += 5
                        elif counter == 3:
                            points += 2
                        else:
                            points += 1
        
        self._ui.points.display(str(points))
    
    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            all_infos = json.load(f)
            domain = JSONLookupDomain(all_infos['domain'])
            widget = EditWidget(filename, domain, all_infos['module'])
            widget._test_cases = {}

            for test in all_infos['tests']:
                user_acts = []
                for act in test['acts']:
                    user_act = UserAct()
                    user_act.score = 1.0
                    user_act.type = UserActionType(act['type'])
                    user_act.slot = act['slot']
                    user_act.value = act['value']
                    user_acts.append(user_act)
                widget._test_cases[frozenset(user_acts)] = test['sentences']

            widget._ui.test_cases.set_sentences(
                widget._test_cases.get(widget._curr_user_acts, []),
                widget._curr_user_acts)
            return widget
    
    def save(self):
        all_infos = {
            'domain': str(self._domain.name),
            'module': self._module_name,
            'tests': []
        }
        for user_act_set in self._test_cases:
            acts_json = {
                'acts': [],
                'sentences': self._test_cases[user_act_set]
            }
            all_infos['tests'].append(acts_json)
            for user_act in user_act_set:
                act_json = {
                    'type': str(user_act.type.value),
                    'slot': user_act.slot,
                    'value': user_act.value
                }
                acts_json['acts'].append(act_json)

        with open(self._file_out, 'w') as f:
            json.dump(all_infos, f, indent=2, sort_keys=True)
    
    def _on_user_acts_changed(self, user_acts):
        self._curr_user_acts = frozenset(user_acts)
        sentences = self._test_cases.get(self._curr_user_acts, [])
        self._ui.test_cases.set_sentences(sentences, self._curr_user_acts)

    def _on_test_cases_changed(self, test_cases):
        self._test_cases[self._curr_user_acts] = test_cases
        self._count_points()
        self.save()
