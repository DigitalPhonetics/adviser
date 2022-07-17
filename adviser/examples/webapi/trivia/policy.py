from typing import List, Dict

from utils.domain.lookupdomain import LookupDomain
from services.service import PublishSubscribe, Service
from utils import SysAct, SysActionType
from utils.logger import DiasysLogger
from utils.useract import UserAct, UserActionType
from collections import defaultdict
from utils.beliefstate import BeliefState
from utils import SysAct


class TriviaPolicy(Service):
    def __init__(self, domain: LookupDomain, logger: DiasysLogger = DiasysLogger()):
        # only call super class' constructor
        self.first_turn = True
        Service.__init__(self, domain=domain, debug_logger=logger)

    @PublishSubscribe(
        sub_topics=["beliefstate"],
        pub_topics=["sys_acts", "sys_state"]
    )
    def generate_sys_acts(
            self,
            beliefstate: BeliefState = None,
            sys_act: SysAct = None
        ) -> dict(sys_acts=List[SysAct]):
        sys_state = {}
        
        if self.first_turn and not beliefstate['user_acts']:
            self.first_turn = False
            return {'sys_acts': [SysAct(SysActionType.Welcome)]}
        elif UserActionType.Bad in beliefstate["user_acts"]:
            return { 'sys_acts': [SysAct(SysActionType.Bad)] }
        elif UserActionType.Bye in beliefstate["user_acts"]:
            return { 'sys_acts': [SysAct(SysActionType.Bye)] }
        elif UserActionType.Deny in beliefstate["user_acts"]:
            self.domain.level = 'easy'
            self.domain.quiztype = 'boolean'
            self.domain.category = 'general'
            self.domain.length = '5'
        else:
            if 'level' in beliefstate['informs']:
                self.domain.level = beliefstate['informs']['level']
            if 'quiztype' in beliefstate['informs']:
                self.domain.quiztype = beliefstate['informs']['quiztype']
            if 'category' in beliefstate['informs']:
                self.domain.category = beliefstate['informs']['category']
            if 'length' in beliefstate['informs']:
                self.domain.length = beliefstate['informs']['length']

            if not self.domain.level:
                return {'sys_acts': [SysAct(SysActionType.Customize, slot_values={
                    'slot': 'level'
                })]}
            elif not self.domain.quiztype:
                return {'sys_acts': [SysAct(SysActionType.Customize, slot_values={
                    'slot': 'quiztype'
                })]}
            elif not self.domain.category:
                return {'sys_acts': [SysAct(SysActionType.Customize, slot_values={
                    'slot': 'category'
                })]}
            elif not self.domain.length:
                return {'sys_acts': [SysAct(SysActionType.Customize, slot_values={
                    'slot': 'length'
                })]}

        self.domain.find_entities(beliefstate["informs"])
        
        if beliefstate['requests']:
            if 'true' in beliefstate['requests']:
                given_answer = True
            else:
                given_answer = False

            is_correct = True if given_answer == self.domain.correct_answer else False
            if is_correct:
                self.domain.score += 1
            correct_text = 'correct' if is_correct else 'incorrect'

            sys_act = SysAct(
                SysActionType.TellQuestion, slot_values={
                    'question': self.domain.question,
                    'given_answer': correct_text
                }
            )
        else:
            sys_act = SysAct(
                SysActionType.TellFirstQuestion, slot_values={
                    'question': self.domain.question
                }
            )
        sys_state = {'last_act': sys_act}
        
        self.debug_logger.dialog_turn("System Action: " + str(sys_act))
        if 'last_act' not in sys_state:
            sys_state['last_act'] = sys_act
        return {
            'sys_acts': [sys_act],
            'sys_state': sys_state,
        }
    
