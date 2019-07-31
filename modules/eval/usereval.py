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

from modules.module import Module
from utils import SysAct, SysActionType, Goal, Constraint, DiasysLogger
from utils.domain.jsonlookupdomain import JSONLookupDomain
from modules.policy.evaluation import ObjectiveReachedEvaluator


class Setup:
    def __init__(self, sections):
        assert len(sections) == 5
        self.rl = self.analyse_group(sections[0])
        self.conf_id = self.analyse_id(sections[1])
        self.description = sections[2]
        self.constraints = self.analyse_constraints(sections[3])
        self.requests = self.analyse_requests(sections[4])
        # self.goal = Goal(Domain.ImsModulesImp)
        # self.goal.init(random_goal=False,
        #                constraints=self.constraints,
        #                requests=self.requests)
    
    def analyse_group(self, text):
        assert text.lower() in ('group re', 'group hcp')
        return text.lower() == 'group re'
    
    def analyse_id(self, text):
        conf_id = int(text)
        assert conf_id >= 0
        return conf_id
    
    def analyse_constraints(self, text):
        constraints = []
        for line in text.split('\n'):
            line = line.strip()
            if line == '':
                continue
            assert '=' in line
            sides = line.split('=')
            assert len(sides) == 2
            constraints.append(Constraint(sides[0].strip(), sides[1].strip()))
        return constraints
    
    def analyse_requests(self, text):
        slots = []
        for line in text.split('\n'):
            line = line.strip()
            if line == '':
                continue
            slots.append(line)
        return slots
    
    def sanity_check(self, domain : JSONLookupDomain):
        for constraint in self.constraints:
            if constraint.slot not in domain.get_informable_slots():
                return False, 'Informable slot not found: %s' % constraint.slot
            if constraint.value not in domain.get_possible_informable_values(constraint.slot):
               return False, 'Slot value not found: %s=%s' % (constraint.slot, constraint.value)
        
        for request in self.requests:
            if request not in domain.get_requestable_slots():
                return False, 'Requestable slot not found: %s' % request
        
        return True, 'All good'


def read_config(config_file):
    sections = config_file.read().split('----')
    sections = [s.strip() for s in sections if s.strip() != '']

    assert len(sections) % 5 == 0
    return [Setup(sections[i:i+5]) for i in range(0, len(sections), 5)]


class Evaluation(Module):
    def __init__(self, domain, subgraph: dict = None, logger : DiasysLogger =  DiasysLogger()):
        super(Evaluation, self).__init__(self, domain, logger = logger)
        self.domain = domain
        self.subgraph = subgraph
        self.is_training = False
        self.setup = None
        self.goal = Goal(domain)
        self.checker = ObjectiveReachedEvaluator(domain)
        self.turns = 0
    
    def set_setup(self, setup):
        self.goal.init(random_goal=False, constraints=setup.constraints, requests=setup.requests)
        self.setup = setup

    def forward(self, dialog_graph, sys_act: SysAct, **kwargs) -> dict():
        self.turns += 1

        if sys_act.type in (SysActionType.Inform, SysActionType.InformByAlternatives, SysActionType.InformByName) \
                and sys_act.slot_values.get('name', '') != 'none':
            for slot in sys_act.slot_values:
                self.goal.fulfill_request(slot, sys_act.slot_values[slot])
        _, success = self.checker.get_final_reward(self.goal)
        self.logger.dialog_turn('Success: %s' % success)
        return {}

    def start_dialog(self, **kwargs):
        self.turns = 0

        return {}

    def end_dialog(self, sim_goal):
        _, success = self.checker.get_final_reward(self.goal)

        output = '\n############ Automatic Evaluation Summary #################\n'
        output += 'Tested: %s\n' % ('RL-based Policy' if self.setup.rl else 'Handcrafted Policy')
        output += 'Task:\n\tConstraints:\n'
        for constraint in self.setup.constraints:
            output += '\t\t%s = %s\n' % (constraint.slot, constraint.value)
        output += '\tRequests:\n\t\t%s\n' % '\n\t\t'.join(self.setup.requests)
        output += '\tID: %d\n' % self.setup.conf_id
        output += '\tDescription: %s\n\n' % self.setup.description
        output += 'Automatic evaluation:\n%s\nNumber of turns: %d\n' % ('SUCCESS' if success else 'FAILURE', self.turns)
        output += '#############################################################'

        self.logger.info(output)

