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
            self.domain.category = 'science'
            self.domain.length = '5'
        else:
            constraints = beliefstate['informs']
            if constraints:
                if 'level' in constraints:
                    self.domain.level = constraints['level']
                if 'quiztype' in constraints:
                    self.domain.quiztype = constraints['quiztype']
                if 'category' in constraints:
                    self.domain.category = constraints['category']
                if 'length' in constraints:
                    self.domain.length = constraints['length']

            if not self.domain.level:
                return {
                    'sys_acts': [
                        SysAct(SysActionType.Customize, slot_values={
                            'slot': 'level'
                        })
                    ]
                }
            elif not self.domain.quiztype:
                return {
                    'sys_acts': [
                        SysAct(SysActionType.Customize, slot_values={
                            'slot': 'quiztype'
                        })
                    ]
                }
            elif not self.domain.category:
                return {
                    'sys_acts': [
                        SysAct(SysActionType.Customize, slot_values={
                            'slot': 'category'
                        })
                    ]
                }
            elif not self.domain.length:
                return {
                    'sys_acts': [
                        SysAct(SysActionType.Customize, slot_values={
                            'slot': 'length'
                        })
                    ]
                }
        
        prev_correct_answer = self.domain.correct_answer

        self.domain.find_entities(beliefstate["informs"])

        if not beliefstate['requests']:
            if self.domain.quiztype == 'boolean':
                return {
                    'sys_acts': [
                        SysAct(SysActionType.TellFirstQuestion, slot_values={
                            'question': self.domain.question,
                            'quiztype': 'boolean',
                            'a': None,
                            'b': None,
                            'c': None,
                            'd': None
                        })
                    ]
                }
            else:
                possible_answers = self.domain.correct_answer | self.domain.incorrect_answers
                return {
                    'sys_acts': [
                        SysAct(SysActionType.TellFirstQuestion, slot_values={
                            'question': self.domain.question,
                            'quiztype': 'multiple',
                            'a': possible_answers['a'],
                            'b': possible_answers['b'],
                            'c': possible_answers['c'],
                            'd': possible_answers['d']
                        })
                    ]
                }
        
        if self.domain.quiztype == 'boolean':
            is_correct = True if bool(beliefstate['requests']) == prev_correct_answer else False
        else:
            is_correct = True if beliefstate['requests'] in prev_correct_answer else False

        try:
            int(self.domain.length)
        except ValueError:
            if is_correct:
                if self.domain.quiztype == "boolean":
                    return {
                        "sys_acts" : [
                            SysAct(SysActionType.TellQuestion, slot_values={
                                'question': self.domain.question,
                                'given_answer': None,
                                'quiztype': 'boolean',
                                'length': 'infinity',
                                'a': None,
                                'b': None,
                                'c': None,
                                'd': None
                            })
                        ]
                    }
                else:
                    return {
                        "sys_acts" : [
                            SysAct(SysActionType.TellQuestion, slot_values={
                                'question': self.domain.question,
                                'given_answer': None,
                                'quiztype': 'multiple',
                                'length': 'infinity',
                                'a': possible_answers['a'],
                                'b': possible_answers['b'],
                                'c': possible_answers['c'],
                                'd': possible_answers['d']
                            })
                        ]
                    }
            else:
                if self.domain.quiztype == "boolean":
                    return {
                        "sys_acts": [
                            SysAct(SysActionType.TellEnd, slot_values={
                                'quiztype': 'boolean',
                                'given_answer': None,
                                'correct_answer': None,
                                'length': 'infinity',
                                'score': str(self.domain.score),
                                'count': str(self.domain.count),
                            })
                        ]
                    }
                else:
                    return {
                        "sys_acts": [
                            SysAct(SysActionType.TellEnd, slot_values={
                                'quiztype': 'multiple',
                                'given_answer': None,
                                'correct_answer': None,
                                'length': 'infinity',
                                'score': str(self.domain.score),
                                'count': str(self.domain.count),
                            })
                        ]
                    }       
        else:
            if self.domain.count < int(self.domain.length):
                self.domain.score += 1 if is_correct else 0
                if self.domain.quiztype == "boolean":
                    return {
                        "sys_acts" : [
                            SysAct(SysActionType.TellQuestion, slot_values={
                                'question': self.domain.question,
                                'given_answer': 'correct' if is_correct else 'incorrect',
                                'correct_answer': None,
                                'quiztype': 'boolean',
                                'length': 'number',
                                'a': None,
                                'b': None,
                                'c': None,
                                'd': None
                            })
                        ]
                    }
                else:
                    possible_answers = self.domain.correct_answer | self.domain.incorrect_answers
                    return {
                        "sys_acts" : [
                            SysAct(SysActionType.TellQuestion, slot_values={
                                'question': self.domain.question,
                                'given_answer': 'correct' if is_correct else 'incorrect',
                                'quiztype': 'multiple',
                                'length': 'number',
                                'correct_answer': [
                                    f'{key.capitalize()}) {value}' for key, value in prev_correct_answer.items()
                                ][0] if not is_correct else "None",
                                'a': possible_answers['a'],
                                'b': possible_answers['b'],
                                'c': possible_answers['c'],
                                'd': possible_answers['d']
                            })
                        ]
                    }
            else:
                if self.domain.quiztype == "boolean": 
                    return {
                        "sys_acts": [
                            SysAct(SysActionType.TellEnd, slot_values={
                                'quiztype': 'boolean',
                                'given_answer': 'correct' if is_correct else 'incorrect',
                                'correct_answer': None,
                                'length': 'number',
                                'score': str(self.domain.score),
                                'count': str(self.domain.count),
                            })
                        ]
                    }
                else:
                    return {
                        "sys_acts": [
                            SysAct(SysActionType.TellEnd, slot_values={
                                'quiztype': 'multiple',
                                'given_answer': 'correct' if is_correct else 'incorrect',
                                'length': 'number',
                                'correct_answer': [
                                    f'{key.capitalize()}) {value}' for key, value in prev_correct_answer.items()
                                ][0] if not is_correct else "None",
                                'score': str(self.domain.score),
                                'count': str(self.domain.count),
                            })
                        ]
                    }

        sys_state = {'last_act': sys_act}
        
        self.debug_logger.dialog_turn("System Action: " + str(sys_act))
        
        return {
            'sys_acts': [sys_act],
            'sys_state': sys_state,
        }
