###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
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

"""The console module provides ADVISER services for tracking current domain"""

from services.service import PublishSubscribe
from services.service import Service
from utils.domain import Domain
from typing import List


class DomainTracker(Service):
    """
        Responsible for selecting which domain should be active at a given time.
        Current implmentation uses keywords to switch domains.
    """

    def __init__(self, domains: List[Domain], greet_on_first_turn: bool = False):
        Service.__init__(self, domain="")
        self.domains = domains
        self.current_domain = None
        self.greet_on_first_turn = greet_on_first_turn

    def dialog_start(self):
        """
            Resets the domain tracker for the start of a new dialog
        """
        self.turn = 0
        self.current_domain = None

    @PublishSubscribe(sub_topics=["gen_user_utterance"], pub_topics=["user_utterance", "sys_utterance"])
    def select_domain(self, gen_user_utterance: str = None) -> dict(user_utterance=str):
        """
            Determines which domain should currently be active. In general, if a keyword is mentioned, the domain
            will change, otherwise it is assumed that the previous domain is still active.

            Args:
                gen_user_utterance (str): the user utterance, before a domain has been determined

            Returns:
                (dict): A dictionary with "user_utterane" as a key and a string as the value with the
                        selected domain appended to the end so the message can be properly routed.
        """

        self.turn += 1
        if self.turn == 1 and self.greet_on_first_turn:
            return {'sys_utterance': "Hello, please let me know how I can help you, I can discuss " +
                    f"the following domains: {self.domains_to_str()}."}

        # if there is only a single domain, simply route the message forward
        if len(self.domains) == 1:
            self.current_domain = self.domains[0]

        # make sure the utterance is lowercase if there is one
        user_utterance = gen_user_utterance
        if user_utterance:
            user_utterance = gen_user_utterance.strip().lower()

        # perform keyword matching to see if any domains are explicitely made active
        active_domains = [d for d in self.domains if d.get_keyword() in user_utterance]

        # Even if no domain has been specified, we should be able to exit
        if "bye" in user_utterance and not self.current_domain:
            return {"sys_utterance": "Thank you, goodbye."}

        # if there are active domains, use the first one
        elif active_domains:
            out_key = f"user_utterance/{active_domains[0].get_domain_name()}"
            self.current_domain = active_domains[0]
            return {out_key: user_utterance}

        # if no domain is explicitely mentioned, assume the last one is still active
        elif self.current_domain:
            out_key = f"user_utterance/{self.current_domain.get_domain_name()}"
            return {out_key: user_utterance}

        # Otherwise ask the user what domain they want
        else:
            return {"sys_utterance": "Hello, please let me know how I can help you, I can discuss " +
                    f"the following domains: {self.domains_to_str()}."}

    def domains_to_str(self):
        """
            Method to create the greeting on the first turn, grammatically joins the names of possible domains into
            a string

            Returns:
                (str): String representing a list of all domain names the system can talk about
        """
        if len(self.domains) == 1:
            return self.domains[0].get_display_name()
        elif len(self.domains) == 2:
            return " and ".join([d.get_display_name() for d in self.domains])
        else:
            return ", ".join([d.get_display_name() for d in self.domains][:-1]) + f", and {self.domains[-1].get_display_name()}"
