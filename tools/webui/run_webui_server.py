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
import uuid
import sys
import time
import copy
import _pickle as cPickle
import zlib

from flask import Flask, session, request, jsonify
from flask_jwt_extended import JWTManager
from flask_cors import CORS, cross_origin


head_location = os.path.realpath(os.curdir)
end = head_location.find('adviser')
head_location = head_location[:end]
sys.path.append(head_location + "/adviser")

from utils.beliefstate import BeliefState
from dialogsystem import DialogSystem
from modules.nlu import HandcraftedNLU
from modules.bst import HandcraftedBST
from modules.nlg import HandcraftedNLG
from modules.policy import DQNPolicy, HandcraftedPolicy
from utils import DiasysLogger
from utils.sysact import SysActionType
from utils.useract import UserActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.common import Language



app = Flask(__name__)
app.secret_key = os.urandom(16)
app.config['JWT_COOKIE_DOMAIN'] = '.adviser.local'
CORS(app, supports_credentials=True)
jwt = JWTManager(app)


class TurnDiff(object):
    def __init__(self, module_idx, module_name, diff_dict):
        self.module_idx = module_idx
        self.module_name = module_name
        self.diff_dict = diff_dict

    def to_dict(self):
        return {
            "index": self.module_idx,
            "name": self.module_name,
            "diff": self.diff_dict
        }
        

ACTIVE_LANGUAGE = Language.ENGLISH
logger = DiasysLogger()
# TODO make domain configurable via command line parameters
domain = JSONLookupDomain('ImsCourses')
nlu = HandcraftedNLU(domain=domain, logger=logger, language=ACTIVE_LANGUAGE)
bst = HandcraftedBST(domain=domain, logger=logger)
# policy = DQNPolicy(domain=domain, logger=self.logger)
policy = HandcraftedPolicy(domain=domain, logger=logger)
# policy.load()
policy.eval()
nlg = HandcraftedNLG(domain=domain, logger=logger, language=ACTIVE_LANGUAGE)


class DialogSystemWrapper(DialogSystem):
    SESSIONS = {}
    
    def __init__(self, uid):
        
        super().__init__(nlu, bst, policy, nlg,
                         domain=domain,
                         logger=logger)
        self.kwargs = self._start_dialog() # run pre-dialog routines
        
        # do one turn without user action to show welcome msg from policy
        self.kwargs, _ = super(DialogSystemWrapper, self)._forward_turn(self.kwargs)
        session['sys_act'] = self.kwargs['sys_utterance']

        self.init_time = time.time()
        session['turn_info'] = zlib.compress(cPickle.dumps({}))

        DialogSystemWrapper.SESSIONS[uid] = self

        print("Server running!")


    def _start_dialog(self, kwargs: dict = None):
        kwargs = kwargs or {}
        for module in self.modules:
            result_dict = module.start_dialog(**kwargs)
            kwargs = {**kwargs, **result_dict}
        return kwargs


    @staticmethod
    def get(uid):
        return DialogSystemWrapper.SESSIONS[uid]


    def run_turn(self, user_action):
        self.kwargs['language'] = ACTIVE_LANGUAGE
        self.kwargs.update({'user_utterance': user_action})
        new_kwargs, _ = self.forward_turn(self.kwargs)
        self.kwargs = new_kwargs
        session['sys_act'] = self.kwargs['sys_utterance']


    def _beliefstate_values(self, val_dict, new_state):
        # note: start with top-level dict for states
        for key, val in new_state.items():
            if isinstance(val, dict):
                # found nested dict: recursion
                if not key in val_dict:
                    val_dict[key] = {}
                self._beliefstate_values(val_dict[key], new_state[key])
                if not val_dict[key]:
                    del val_dict[key]
            elif isinstance(val, list):
                if key == 'db_matches':
                    # replace binary match indicators with number
                    count_str = ''
                    if val[0] == True:
                        count_str = '0'
                    elif val[1] == True:
                        count_str = '1'
                    elif val[2] == True:
                        count_str = '2-4'
                    elif val[3] == True:
                        count_str = '> 4'
                    del val_dict[key]
                    val_dict[key] = count_str
                if len(val) == 0:
                    del val_dict[key]
            elif (isinstance(val, float) or isinstance(val, str)) and isinstance(key, str):
                # found key-value pair: compare
                if val == 0.0:
                    del val_dict[key]
                elif val == '**NONE**' or val == 'none':
                    del val_dict[key]
                elif key == '**NONE**' or key == 'none':
                    del val_dict[key]
                else:
                    # Fixes formatting issues for float in the beliefstate
                    if isinstance(val, float):
                        val_dict[key] = "{:1.1f}".format(val)
            elif isinstance(val, type(None)):
                # print("NONE FOUND", key)
                del val_dict[key]
                    

    def forward_turn(self, kwargs):
        """ Forward one turn of a dialog. """
        #self.logger.dialog_turn("# TURN " + str(self.num_turns) + " #")

        # call each module in the list
        turn_dict = []
        for module_idx, module in enumerate(self.modules):
            module_name = type(module).__name__
            module_dict = {}

            # intercept single-module forward output
            result_dict = module.forward(self, **kwargs)

            # store module output in session
            for key, value in result_dict.items():
                if isinstance(value, BeliefState):
                    new_bst = result_dict[key]._history[-1]
                    diff = copy.deepcopy(new_bst)
                    self._beliefstate_values(diff, new_bst)
                    module_dict[key] = diff
                elif key == 'user_acts':
                    # module_dict[key] = [str(usr_act) for usr_act in value]
                    module_dict[key] = [
                        {
                            'type': usr_act.type.value,
                            'slot': usr_act.slot,
                            'value': usr_act.value,
                            'score': "{:1.1f}".format(usr_act.score)
                        } for usr_act in value
                    ]
                elif key == 'sys_act':
                    module_dict[key] = {
                        'type': str(value.type.value),
                        'values': {slot: slotval for slot, slotval in value.slot_values.items()}
                    }
                elif key == 'sys_utterance':
                    module_dict[key] = value
                else:
                    module_dict[key] = result_dict

            # update kwargs
            kwargs = {**kwargs, **result_dict}
            turn_dict.append(TurnDiff(module_idx, module_name, module_dict).to_dict())
        
        # print(turn_dict)
        session['turn_info'] = zlib.compress(cPickle.dumps(turn_dict))

        stop = False
        if 'user_acts' in kwargs and kwargs['user_acts'] is not None:
            for action in kwargs['user_acts']:
                if action.type is UserActionType.Bye:
                    stop = True
            if 'sys_act' in kwargs and kwargs['sys_act'].type == SysActionType.Bye:
                stop = True
    
        self.num_turns += 1
        return kwargs, stop



@app.route('/lang', methods=['GET'])
@cross_origin(supports_credentials=True)
def select_language():
    global ACTIVE_LANGUAGE
    language = request.args.get('lang')
    print("switched language to: ", language)
    if language == "de":
        ACTIVE_LANGUAGE = Language.GERMAN
    elif language == "en":
        ACTIVE_LANGUAGE = Language.ENGLISH 
    return ('', 204)
   


@app.route('/chat', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def main():
    if request.method == 'GET' or 'uid' not in session:
        uid = uuid.uuid4()
        session['dialog'] = ""
        session['uid'] = uid
        DialogSystemWrapper(uid)
        print(f"New session started with uuid {uid}.")
    else:
        user_action = request.json['msg'].strip()
        if user_action.lower() == 'reset' or user_action.lower() == 'restart':
            uid = uuid.uuid4()
            session['dialog'] = ""
            session['uid'] = uid
            DialogSystemWrapper(uid)
            print(f"New session started with uuid {uid}.")
        else:
            system =  DialogSystemWrapper.get(session['uid'])
            system.run_turn(user_action)

    # return dialog as html
    turn_info = cPickle.loads(zlib.decompress(session['turn_info']))
    
    return jsonify({'sys_utterance': session['sys_act'], 
                    'turn_info': turn_info
                    })


    

if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)