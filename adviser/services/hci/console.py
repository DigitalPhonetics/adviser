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

"""The console module provides ADVISER modules that access the console for input and output."""
import math
import os
import time
from typing import Union
from services.service import ControlChannelMessages

from services.service import PublishSubscribe
from services.service import Service
from utils.domain import Domain

import asyncio
from aioconsole import ainput
from services.service import PublishSubscribe, Service


class ConsoleInput(Service):
    """
    Gets the user utterance from the console.

    Waits for the built-in input function to return a non-empty text.
    """
    def __init__(self, identifier="ConsoleInput", domain: Union[str, Domain] = "", conversation_log_dir: str = None, transports: str = "ws://localhost:8080/ws", realm="adviser") -> None:
        super().__init__(identifier=identifier, domain=domain, transports=transports, realm=realm)
        self.conversation_log_dir = conversation_log_dir
        self.waiting = False
    
    async def on_dialog_start(self, user_id: int):
        self.waiting = False # This flag solves the problem of waiting on multiple inputs concurrently when dealing with multiple text input sources, e.g. browser + console

    @PublishSubscribe(sub_topics={ControlChannelMessages.DIALOG_END: "dialog_end"})
    async def turn_end(self, dialog_end: bool):
        # print("CONSOLE START")
        if not dialog_end and not self.waiting:
            self.waiting = True
            await asyncio.create_task(self.get_user_utterance())
        
    @PublishSubscribe(pub_topics=["gen_user_utterance"])
    async def get_user_utterance(self):
        # print(f"WAITING FOR UTTERANCE from {user_id}")
        utterance = await ainput(">>>")
        if self.conversation_log_dir is not None:
            with open(os.path.join(self.conversation_log_dir, (str(math.floor(time.time())) + "_user.txt")), "w") as conv_log:
                conv_log.write(utterance)
        # print(" - got ", utterance)
        self.waiting = False
        return {"gen_user_utterance": utterance}


class ConsoleOutput(Service):
    def __init__(self, identifier="ConsoleOutput", domain: Union[str, Domain] = "", transports: str = "ws://localhost:8080/ws", realm="adviser") -> None:
        super().__init__(identifier=identifier, domain=domain, transports=transports, realm=realm)
        
    @PublishSubscribe(sub_topics=["sys_utterance"], pub_topics=[ControlChannelMessages.DIALOG_END])
    async def print_sys_utterance(self, sys_utterance: str):
        """

        The message is simply printed to the console.

        Args:
            sys_utterance (str): The system utterance

        Returns:
            dict with entry dialog_end: True or False

        Raises:
            ValueError: if there is no system utterance to print
        """
        if sys_utterance is not None and sys_utterance != "":
            print("System: {}".format(sys_utterance))
        else:
            raise ValueError("The system utterance is empty. Did you forget to call an NLG module before?")
        
        # print(f"GOT UTTERANCE", user_utterance)
        return {ControlChannelMessages.DIALOG_END: 'bye' in sys_utterance}

