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

import asyncio

from services.service import PublishSubscribe
from services.service import Service
import webbrowser
import asyncio
import os
class GUIServer(Service):
    def __init__(self, logger=None):
        super().__init__(domain="", identifier="GUIServer")
        self.websocket = None
        self.loopy_loop = asyncio.new_event_loop()
        # open UI in webbrowser automatically
        webui_path = f"file:///{os.path.join(os.path.realpath(''), 'tools', 'webui', 'chat.html')}"
        print("WEBUI accessible at", webui_path)
        webbrowser.open(webui_path)

    @PublishSubscribe(pub_topics=['gen_user_utterance'])
    def user_utterance(self, message = ""):
        return {'gen_user_utterance': message}

    @PublishSubscribe(sub_topics=['sys_utterance'])
    def forward_sys_utterance(self, sys_utterance: str):
        self.forward_message_to_react(message=sys_utterance, topic="sys_utterance")

    def forward_message_to_react(self, message, topic: str):
        asyncio.set_event_loop(self.loopy_loop)
        if self.websocket:
            self.websocket.write_message({"topic": topic, "msg": message})
