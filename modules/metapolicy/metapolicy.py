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

# from typing import List

from dialogsystem import DialogSystem
from utils.logger import DiasysLogger
from utils.sysact import SysActionType
from utils.useract import UserActionType
# from utils.domain.jsonlookupdomain import JSONLookupDomain


class HandcraftedMetapolicy(DialogSystem):
    """ Base class for handcrafted meta policies; inherits from DialogSystem so
    it can control the flow of information between single domain dialogs.

    Provides a simple rule-based meta policy for switching between and combining
    domains in a multidomain dialog system. This is accomplished through rewrite
    and combination rules, where queries are rewritten if they were phrased in
    an ambiguous way (one which uses abstract referants such as 'this' or 'that')
    which cannot be resolved in a single domain. Rewriting involves replacing each
    ambiguous reference with a concrete one (eg. 'this course' -> 'Ethics in NLP')
    and then sending the rewritten query back through the processing pipeline.
    Combination rules determine which single domain outputs should be passed on
    to the user based on which domains are currently active.

    In addition to the rewrite and combination rules, the handcrafted meta policy
    is responsible for coordinating a single domain dialog for each of the domains
    it is initialized with. This functionality is largely the same as what a
    DialogSystem does with the notable difference that a single dialog turn for
    the meta policy involves stepping all domains through a single turn, rather
    than only one.

    In order to create your own meta policy, you can inherit from this class.
    Make sure to overwrite the `forward_turn`-method with whatever additionally
    rules/functionality required.


    """
    def __init__(self, subgraphs, in_module, out_module, logger: DiasysLogger = DiasysLogger()):
        domains = [graph.domain.get_domain_name() for graph in subgraphs]
        super(HandcraftedMetapolicy, self).__init__(domains, None, logger=logger)
        self.subgraphs = subgraphs
        self.active_domains = domains
        self.input_module = in_module
        self.output_module = out_module
        self.rewrite = False
        self.offers = {}  # track offers made for active domains

    def _start_dialog(self, kwargs: dict = None):
        # start of dialog
        kwarg_dic = {graph.domain.get_domain_name(): {} for graph in self.subgraphs}
        self.logger.dialog_turn("# DIALOG {} STARTED #".format(self.num_dialogs+1))
        for graph in self.subgraphs:
            for module in graph.modules:
                kwargs = kwarg_dic[graph.domain.get_domain_name()]
                kwarg_dic[graph.domain.get_domain_name()] = {
                    **kwargs, **module.start_dialog(**kwargs)}
        return kwarg_dic

    def run_dialog(self, max_length=-1):
        """Method responsible for coordinating  an entire dialog, takes an
           optional maximum number of turns and coordinates each turn until
           the user either types 'bye' or the maximum number of turns has
           been reached

        Args:
            max_length (int): the maximum number of dialog turns; default is -1

        --LV
        """

        # start of dialog
        kwarg_dic = {graph.domain.get_domain_name(): {} for graph in self.subgraphs}
        self.logger.dialog_turn("# DIALOG {} STARTED #".format(self.num_dialogs+1))
        kwarg_dic = self._start_dialog()

        idx_turn = 0
        self.num_turns = 0
        stop = False
        while True:
            if idx_turn == max_length:
                break
            idx_turn += 1

            kwarg_dic = self._forward_turn(kwarg_dic)

            # check each graph for a 'bye' act --LV
            for graph in kwarg_dic:
                kwargs = kwarg_dic[graph]
                if 'user_acts' in kwargs and kwargs['user_acts'] is not None:
                    stop = False
                    for action in kwargs['user_acts']:
                        if action.type is UserActionType.Bye:
                            stop = True
                if 'sys_act' in kwargs \
                        and kwargs['sys_act'].type == SysActionType.Bye:
                    stop = True
            if stop:
                break

        # TODO we need a different user simulator for meta policy?
        if 'sim_goal' in kwargs:
            sim_goal = kwargs['sim_goal']
        else:
            sim_goal = None

        # end of dialog
        for graph in self.subgraphs:
            for module in graph.modules:
                module.end_dialog(sim_goal)

        self.logger.dialog_turn("# DIALOG {} FINISHED #".format(self.num_dialogs+1))
        self.num_dialogs += 1

    def _forward_turn(self, kwarg_dic):
        """Responsible for stepping each subgraph one turn and combining/rewriting the
           outputs as necessary. In the case of a rewrite, the graph will take
           another turn without giving/requesting user input; using instead the
           rewritten user utterance

        Args:
            kwarg_dic (dict): dictionary where keys are the domain names and
            values are dictionaries of the arguments needed to call the forward
            method for each graph

        Returns:
            (dict): the updated kwarg_dic after a turn

        --LV
        """
        self.sys_utterances = {}
        self.sys_act = {}
        self.user_acts = {}
        bad_act_text = ""
        user_utterance = None

        language = None
        # check if we should get a new user utterance or use a rewritten one
        if not self.rewrite:
            user_utterance = self.input_module.forward(self)
            # on the first turn, there won't be a user utterance
            if user_utterance:
                # given as a dict, so we want just the value
                language = user_utterance['language']
                user_utterance = user_utterance['user_utterance']
            else:
                user_utterance = ''

        for graph in self.subgraphs:
            kwargs = kwarg_dic[graph.domain.get_domain_name()]
            kwargs['language'] = language
            # if we rewrote it, the correct value will be there already --LV
            if not self.rewrite:
                kwargs['user_utterance'] = user_utterance

            # step one turn through each graph --LV
            kwargs, stop = graph._forward_turn(kwargs)
            kwarg_dic[graph.domain.get_domain_name()] = kwargs

            # Not sure if we need all of these explicitely, might change later --LV
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

        # If there is no valid sys_act, try rewriting user input. If it
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
            self.output_module.forward(self, sys_utterance=output)

        self.num_turns += 1

        return kwarg_dic

    def _update_current_offers(self):
        """Helper function; when an offer is made, update the offers dictionary for
           that domain, so the metapolicy can disambiguate "this" or "it" to
           the most recent offer

        --LV
        """

        # for each subgraph if domain is active + offer; store value --LV
        for graph in self.subgraphs:
            d = graph.domain.get_domain_name()
            if d in self.active_domains:
                sys_act = self.sys_act[d]
                if sys_act.type == SysActionType.Inform or\
                        sys_act.type == SysActionType.InformByName or\
                        sys_act.type == SysActionType.InformByAlternatives:

                    # TODO: is it ever going to be possible to have multiple
                    # offers? If so need to fix this --LV
                    # prim_key = graph.domain.get_primary_key()
                    # current_offer = sys_act.get_values(prim_key)[0]
                    self.offers[d] = sys_act

    def _update_active_domains(self):
        """Helper function; checks which domains are likely to be active based
           on which return something other than a BadAct, updates the active
           domains dictionary.

           This is especially important for the case of rewrite. The correct
           domain is generally recognized as active with an action of
           informbyname(name=none) act before the rewrite, so in the case that
           after the rewrite multiple domains are active, updating the active
           domains prioritizes the correct one, since it prioritzes any that were
           also active in the previous turn.

        --LV
        """
        current_active_domains = []
        # If a graph was active last turn, assume it's still active this turn
        for graph in self.sys_utterances:
            if graph in self.active_domains:
                current_active_domains.append(graph)
        # If none of the previous domains are active, but there are outputs for
        # other domains, assume they are now relevant
        if not current_active_domains:
            for graph in self.sys_utterances:
                current_active_domains.append(graph)
        self.active_domains = current_active_domains

    def _combine_outputs(self):
        """Chooses between/combines system utterances to give the user a
           single repsonse

        Returns:
            (str): output string from the system

        """
        inform_types = [SysActionType.InformByAlternatives,
                        SysActionType.InformByName]
        if self.num_turns == 0:
            return self._gen_greeting()
        valid_output = []
        informs = []
        requests = []
        # collect all valid informs and requests
        for graph in self.subgraphs:
            d = graph.domain.get_domain_name()
            prim_key = graph.domain.get_primary_key()
            if d in self.active_domains:
                out = self.sys_utterances[d]
                if out not in valid_output:
                    valid_output.append(out)
                    sys_act = self.sys_act[d]
                    if sys_act.type in inform_types:
                        if sys_act.get_values(prim_key) and (
                                sys_act.get_values(prim_key) != ['none']):
                            informs.append(out)
                    if sys_act.type == SysActionType.Request:
                        if not sys_act.get_values(prim_key):
                            requests.append(out)
        # Inform acts take precedence over all other act types
        if informs:
            valid_output = informs
        elif requests:
            valid_output = requests
        output = " And ".join(valid_output)
        return output

    def _rewrite_user_input(self, user_utterance):
        """If one of the rewrite rules matches, rewrite the user input following
           that rule; additionally set the rewrite flag to true

        Args:
            usr_utterance (str): string representing what the user said (or
                                 previous rewrite)

        Returns:
            (dict): dictionary where keys are graph domains and values are the
                    rewritten uesr utterance for each domain

        --LV
        """
        out_dic = {}

        #                     ### RULES ###

        # naive rule, if 'this' in query, replace with last offer
        # TODO: replace with regex at some point --LV

        for graph in self.subgraphs:
            if graph.domain.get_domain_name() in self.offers:
                offer = self.offers[graph.domain.get_domain_name()]
                for slot in offer.slot_values.keys():
                    pronouns = graph.domain.get_pronouns(slot)
                    for pronoun in pronouns:
                        user_utterance = user_utterance.replace(pronoun, offer.get_values(slot)[0])

                # TODO: look into more complex cases eg. different rewrites for
                # different graphs --LV
                for graph in self.subgraphs:
                    out_dic[graph.domain.get_domain_name()] = user_utterance
                self.rewrite = True
        return out_dic

    def _gen_greeting(self):
        """Function to generate the greeting based on what subraphs are present so
           the user knows which domains are represented

        Returns:
            (str): string representing the greeting; listing all supported
                   domains

        --LV
        """
        supported_domains = [graph.domain.get_domain_name() for graph in self.subgraphs]
        output = "Welcome to the IMS multi-domain chat bot. " +\
                 "Please let me know what you are looking for. " +\
                 "I can help you find "
        if len(supported_domains) == 1:
            output += supported_domains[0] + "."
        elif len(supported_domains) == 2:
            output += " and ".join(supported_domains) + "."
        elif len(supported_domains) > 2:
            output += ", ".join(supported_domains[:-1]) + ", and "
            output += supported_domains[-1] + "."
        else:
            output = "I'm sorry there are currently no domains specified."

        return output
