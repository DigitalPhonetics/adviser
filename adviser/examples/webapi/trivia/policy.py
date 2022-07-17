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
        
        self.prev_sys_act = sys_act
        sys_state = {}

        if self.first_turn and not beliefstate['user_acts']:
            self.first_turn = False
            return {'sys_acts': [SysAct(SysActionType.Welcome)]}
        elif UserActionType.Bad in beliefstate["user_acts"]:
            return { 'sys_acts': [SysAct(SysActionType.Bad)] }
        elif UserActionType.Bye in beliefstate["user_acts"]:
            return { 'sys_acts': [SysAct(SysActionType.Bye)] }
        else:
            entities_constraints = {}
            self.domain.find_entities(beliefstate["informs"])
            
            if beliefstate['requests']:
                try: 
                    beliefstate['requests']['true']
                except KeyError:
                    given_answer = False
                else:
                    given_answer = True
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
    
