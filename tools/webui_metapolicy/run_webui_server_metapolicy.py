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
from utils.sysact import SysActionType, SysAct
from utils.useract import UserActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.common import Language
from modules.metapolicy.metapolicy import HandcraftedMetapolicy


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
        

class MutliDomainDialogSystem(object):
    """This is the main dialog system, holding all modules and taking care of
    the data flow.

    Public methods:
    train -- trains the dialog system
    chat -- allows to chat with the dialog system
    eval -- Evaluates the dialog system

    Instance variables:
    current_domain -- the currently active domain
    modules -- a list of modules which will be called in the given order
    sequentially
    """

    def __init__(self, *modules, domain=None, logger: DiasysLogger = DiasysLogger()):
        self.domain = domain
        self.logger = logger
        self.modules = modules

        self.is_training = False
        self.num_dialogs = 0
        self.num_turns = 0

    def forward_turn(self, kwargs):
        """ Forward one turn of a dialog. """

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
                            'score': "{:1.1f}".format(usr_act.score),
                            'negate': usr_act.negate
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

        stop = False
        if 'user_acts' in kwargs and kwargs['user_acts'] is not None:
            for action in kwargs['user_acts']:
                if action.type is UserActionType.Bye:
                    stop = True
            if 'sys_act' in kwargs and kwargs['sys_act'].type == SysActionType.Bye:
                stop = True
            elif 'sys_act' in kwargs and kwargs['sys_act'].type == SysActionType.Restart:
                # TODO maybe ignore stop if system wants to restart?
                kwargs = {}
                self.num_turns = 0
                for module in self.modules:
                    kwargs = {**kwargs, **module.start_dialog(**kwargs)}
        self.num_turns += 1
        return kwargs, stop, turn_dict

    def train(self):
        """ Configure all modules in the dialog graph for training mode. """
        self.is_training = True

        # set train flag in each module
        for module in self.modules:
            module.train()

    def eval(self):
        """ Configure all modules in the dialog graph for evaluation mode. """
        self.is_training = False

        # set train flag in each module
        for module in self.modules:
            module.eval()

    def _beliefstate_values(self, val_dict, new_state):
        # note: start with top-level dict for states
        for key, val in new_state.items():
            if isinstance(val, dict):
                # found nested dict: recursion
                if key not in val_dict:
                    val_dict[key] = {}
                self._beliefstate_values(val_dict[key], new_state[key])
                if not val_dict[key]:
                    del val_dict[key]
            elif isinstance(val, list):
                if key == 'db_matches':
                    # replace binary match indicators with number
                    count_str = ''
                    if val[0] is True:
                        count_str = '0'
                    elif val[1] is True:
                        count_str = '1'
                    elif val[2] is True:
                        count_str = '2-4'
                    elif val[3] is True:
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
                del val_dict[key]


ACTIVE_LANGUAGE = Language.ENGLISH
logger = DiasysLogger()
# TODO make domain configurable via command line parameters
domain1 = JSONLookupDomain('ImsLecturers')
l_nlu = HandcraftedNLU(domain=domain1, logger=logger)
l_bst = HandcraftedBST(domain=domain1, logger=logger)
l_policy = HandcraftedPolicy(domain=domain1, logger=logger)
l_nlg = HandcraftedNLG(domain=domain1, logger=logger)
lecturers = MutliDomainDialogSystem(
                        l_nlu,
                        l_bst,
                        l_policy,
                        l_nlg,
                        domain=domain1,
                        logger=logger
)
domain2 = JSONLookupDomain('ImsCourses')
c_nlu = HandcraftedNLU(domain=domain2, logger=logger)
c_bst = HandcraftedBST(domain=domain2, logger=logger)
c_policy = HandcraftedPolicy(domain=domain2, logger=logger)
c_nlg = HandcraftedNLG(domain=domain2, logger=logger)
courses = MutliDomainDialogSystem(
                    c_nlu,
                    c_bst,
                    c_policy,
                    c_nlg,
                    domain=domain2,
                    logger=logger
)
subgraphs = [courses, lecturers]
multi = HandcraftedMetapolicy(
    subgraphs=subgraphs,
    in_module=None, out_module=None,
    logger=logger)


class DialogSystemWrapper(HandcraftedMetapolicy):
    SESSIONS = {}

    def __init__(self, uid, metapolicy):
        super(DialogSystemWrapper, self).__init__(subgraphs=metapolicy.subgraphs,
                                                  in_module=metapolicy.input_module,
                                                  out_module=metapolicy.output_module,
                                                  logger=logger)

        # do one turn without user action to show welcome msg from policy
        self.kwargs = {}
        self.kwargs = self._start_dialog(self.kwargs)  # run pre-dialog routines
        self.kwargs['user_utterance'] = ''
        self.kwargs, _ = self.forward_turn(self.kwargs)
        self.num_turns = 1
        session['sys_act'] = self.kwargs['sys_utterance']

        self.init_time = time.time()
        session['turn_info'] = zlib.compress(cPickle.dumps({}))

        DialogSystemWrapper.SESSIONS[uid] = self

        print("Server running!")

    @staticmethod
    def get(uid):
        return DialogSystemWrapper.SESSIONS[uid]

    def forward_turn(self, kwarg_dic):
        self.sys_utterances = {}
        self.sys_act = {}
        self.user_acts = {}
        bad_act_text = ""
        user_utterance = None

        # check if we should get a new user utterance or use a rewritten one
        user_utterance = kwarg_dic['user_utterance']
        turn_infos = {}

        for graph in self.subgraphs:
            kwargs = kwarg_dic[graph.domain.get_domain_name()]
            kwargs['language'] = ACTIVE_LANGUAGE

            # if we rewrote it, the correct value will be there already --LV

            if not self.rewrite:
                kwargs['user_utterance'] = user_utterance

            # step one turn through each graph --LV
            kwargs, stop, turn_info = graph.forward_turn(kwargs)
            turn_infos[graph.domain.get_domain_name()] = turn_info
            kwarg_dic[graph.domain.get_domain_name()] = kwargs

            # Not sure if we need all of these explicitely, might change later
            # --LV
            if 'user_acts' in kwargs and kwargs['user_acts']:
                self.user_acts[graph.domain.get_domain_name()] = kwargs['user_acts']
            if 'sys_act' in kwargs and kwargs['sys_act']:
                self.sys_act[graph.domain.get_domain_name()] = kwargs['sys_act']
            if 'sys_utterance' in kwargs and kwargs['sys_utterance']:
                if self.sys_act[graph.domain.get_domain_name()].type != SysActionType.Bad:
                    self.sys_utterances[graph.domain.get_domain_name()] = \
                        kwargs['sys_utterance']
                else:
                    # TODO: make sure there aren't different texts later for
                    # different domains --LV
                    bad_act_text = kwargs['sys_utterance']

        self._update_active_domains()
        self._update_current_offers()

        # combine outputs as needed --LV
        output = self._combine_outputs()

        none_inform = True
        for d in self.active_domains:
            sys_act = self.sys_act[d]
            if sys_act.get_values("name") != ['none']:
                none_inform = False

        # If there is no valid sys_act, try combining user input only if it
        # fails, output a BadAct --LV
        if not self.active_domains and user_utterance:
            rewrites = self._rewrite_user_input(user_utterance)
            if rewrites:
                for graph in rewrites:
                    kwarg_dic[graph]['user_utterance'] = rewrites[graph]
            else:
                output = bad_act_text
        elif not self.rewrite and none_inform:
            rewrites = self._rewrite_user_input(user_utterance)
            if rewrites:
                for graph in rewrites:
                    kwarg_dic[graph]['user_utterance'] = rewrites[graph]
        else:
            # If we are clearly not rewriting; reset this flag --LV
            self.rewrite = False

        # log the output (if we rewrite though don't show it to the user) --LV
        self.logger.dialog_turn(output)
        if not self.rewrite:
            self.kwargs['sys_utterance'] = output
        else:
            self.forward_turn(kwarg_dic)
        session['turn_info'] = zlib.compress(cPickle.dumps(turn_infos))
        return kwarg_dic, False

    def run_turn(self, user_action):
        self.kwargs['language'] = ACTIVE_LANGUAGE
        self.kwargs.update({'user_utterance': user_action})
        new_kwargs, _ = self.forward_turn(self.kwargs)
        self.kwargs = new_kwargs
        session['sys_act'] = self.kwargs['sys_utterance']
        self.num_turns += 1


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
        DialogSystemWrapper(uid, multi)
        print(f"New session started with uuid {uid}.")
    else:
        user_action = request.json['msg'].strip()
        if user_action.lower() == 'reset' or user_action.lower() == 'restart':
            uid = uuid.uuid4()
            session['dialog'] = ""
            session['uid'] = uid
            DialogSystemWrapper(uid, multi)
            print(f"New session started with uuid {uid}.")
        else:
            system = DialogSystemWrapper.get(session['uid'])
            system.run_turn(user_action)

    # return dialog as html
    turn_info = cPickle.loads(zlib.decompress(session['turn_info']))

    return jsonify({'sys_utterance': session['sys_act'],
                    'turn_info': turn_info
                    })


if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)
