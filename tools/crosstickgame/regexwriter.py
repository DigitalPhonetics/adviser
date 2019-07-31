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
from typing import List, Tuple, Set

from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QTableWidgetItem, QHeaderView

from modules.nlu.nlu import HandcraftedNLU
from utils import UserAct, UserActionType
from utils.common import Language
from utils.domain.jsonlookupdomain import JSONLookupDomain
from tools.crosstickgame.nlumodules import get_module, update_modules
from tools.regextemplates.rules.regexfile import RegexFile
from tools.regextemplates.templatestojson import create_json_from_template


class _Analysis:
    def __init__(self, module: HandcraftedNLU, sentence: str, expected_user_acts: Set[UserAct]):
        self.module = module
        self.sentence = sentence
        self.expected = expected_user_acts
        self.predicted = self._apply_module()
    
    def is_correct(self) -> bool:
        return self.expected == self.predicted
    
    def _apply_module(self) -> Set[UserAct]:
        result = frozenset()
        try:
            result = self.module.forward(None, user_utterance=self.sentence, language=Language.ENGLISH)['user_acts']
        except BaseException as e:
            print(e)
            raise e
        return frozenset(result)


class RegexWriter(QWidget):
    """TODO"""

    def __init__(self, template_filename: str, test_case_filename: str,
                 domain: JSONLookupDomain, module_name: str, parent: QWidget = None):
        QWidget.__init__(self, parent)
        self._ui = uic.loadUi(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui/regexwriter.ui'), self)

        self._domain = domain
        self._module_name = module_name
        self._module = get_module(domain, module_name)
        self._template_filename = template_filename
        self._test_case_filename = test_case_filename
        self._template = RegexFile(template_filename, domain)
        self._test_cases = []

        self._ui.save_button.clicked.connect(self.save)
        self._ui.error_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._ui.error_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self._load_test_cases()
        self._load_template()
        self._update()
    
    def _load_template(self):
        with open(self._template_filename, 'r') as f:
            content = f.read()
            self._ui.template_content.setPlainText(content)
    
    def _load_test_cases(self):
        with open(self._test_case_filename, 'r') as f:
            all_infos = json.load(f)
            domain = JSONLookupDomain(all_infos['domain'])
            self.test_cases = {}

            for test in all_infos['tests']:
                user_acts = []
                for act in test['acts']:
                    user_act = UserAct()
                    user_act.score = 1.0
                    user_act.type = UserActionType(act['type'])
                    user_act.slot = act['slot']
                    user_act.value = act['value']
                    user_acts.append(user_act)
                self.test_cases[frozenset(user_acts)] = test['sentences']

    def _calculate_accuracy(self):
        all_analyses = self._analyse_all_sentences()
        correct_analyses = [a for a in all_analyses if a.is_correct()]
        return 1.0 * len(correct_analyses) / len(all_analyses)
    
    def _analyse_all_sentences(self) -> List[_Analysis]:
        analyses: List[_Analysis] = []
        for user_act_set in self.test_cases:
            for sentence in self.test_cases[user_act_set]:
                analyses.append(_Analysis(self._module, sentence, user_act_set))
        return analyses
    
    def _pretty_print_user_act(self, user_act: UserAct) -> str:
        if user_act.slot is None:
            return f'{user_act.type.value}()'
        if user_act.value is None:
            return f'{user_act.type.value}({user_act.slot})'
        return f'{user_act.type.value}({user_act.slot}={user_act.value})'
    
    def _pretty_print_user_act_set(self, user_act_set: Set[UserAct]) -> str:
        if not user_act_set:
            return 'no analysis found'
        return '\n'.join([self._pretty_print_user_act(user_act) for user_act in user_act_set])

    def _print_erroneous_sentences(self):
        all_analyses = self._analyse_all_sentences()
        incorrect_analyses = [a for a in all_analyses if not a.is_correct()]
        self._ui.error_table.setRowCount(len(incorrect_analyses))
        for row, incorrect_analysis in enumerate(incorrect_analyses):
            text_for_predicted = self._pretty_print_user_act_set(incorrect_analysis.predicted)
            text_for_expected = self._pretty_print_user_act_set(incorrect_analysis.expected)
            # QTableWidgetItem(text_for_predicted).set
            self._ui.error_table.setItem(row, 0, QTableWidgetItem(text_for_predicted))
            self._ui.error_table.setItem(row, 1, QTableWidgetItem(text_for_expected))
            self._ui.error_table.setItem(row, 2, QTableWidgetItem(incorrect_analysis.sentence))

    def _update(self):
        self._load_test_cases()
        self._ui.accuracy.setText('Accuracy: %.1f%%' % (self._calculate_accuracy() * 100))
        self._print_erroneous_sentences()

    def _analyse_sentence(self, user_acts, sentence):
        result = self._module.forward(None, user_utterance=sentence, language=Language.ENGLISH)['user_acts']
        analysed_user_acts = frozenset(result)
        return _Analysis(self._module, sentence, user_acts).is_correct()

    def save(self):
        with open(self._template_filename, 'w') as f:
            f.write(self._ui.template_content.toPlainText())
        create_json_from_template(self._domain, self._template_filename)
        update_modules()
        self._module = get_module(self._domain, self._module_name)
        self._update()
