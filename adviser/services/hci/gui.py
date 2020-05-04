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

import json
import os
import re
from typing import List
import requests
import socket
import asyncio
# import websockets
import time
import itertools
import sys

def get_root_dir():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(get_root_dir())

from services.service import PublishSubscribe
from services.service import Service
from utils import UserAct, UserActionType
from utils.beliefstate import BeliefState
from utils.common import Language
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger
from utils.sysact import SysAct, SysActionType
from utils.topics import Topic


from flask_socketio import SocketIO
from flask import Flask, render_template, request
import threading
import queue as Queue
import functools


class GUIServer(Service):
    """
    Service for the React-based Web-UI.
    Run this as a remote service:
        * run this file seperately, will start the GUI Server
        * run the dialog system in another python instance, add a RemoteService with identifier `GUIServer`
    """

    def __init__(self, socketio, identifier="GUIServer", logger: DiasysLogger = None):
        super().__init__(domain='', identifier=identifier)
        self.socketio = socketio
        self.logger = logger

    @PublishSubscribe(pub_topics=[Topic.DIALOG_END])
    def start_dialog(self, message: str = ""):
        # TODO logging instead of printing
        print("SERVICE", message)
        return {Topic.DIALOG_END: message}

    @PublishSubscribe(sub_topics=['gen_user_utterance'])
    def forward_message_to_react(self, gen_user_utterance: str = ""):
        print("FWD to react", gen_user_utterance)
        self.socketio.emit('user_utterance', gen_user_utterance)
        
    @PublishSubscribe(sub_topics=['emotion'])
    def forward_message_to_react(self, emotion = None):
        print(f"Got emotion from dialog system: {emotion}")
        emotion["category"] = emotion["category"].value
        probs = emotion["cateogry_probabilities"]
        formatted_probs = {"Angry": probs[0]*100, "Happy": probs[1]*100, "Neutral": probs[2]*100, "Sad": probs[3]*100}
        emotion["cateogry_probabilities"] = formatted_probs
        emotion = json.dumps(emotion)
        self.socketio.emit('emotion', emotion)

    @PublishSubscribe(sub_topics=['engagement'])
    def forward_message_to_react(self, engagement = None):
        print(f"Got engagement from dialog system: {engagement}")
        self.socketio.emit('engagement', engagement.value)

    @PublishSubscribe(pub_topics=['gen_user_utterance'])
    def user_utterance(self, message = ""):
        print(f"SERVICE Message from React: {message}")
        # forward_message_from_react('gen_user_utterance', message)
        return {'gen_user_utterance': message}


    @PublishSubscribe(sub_topics=['sys_utterance'])
    def forward_message_to_react(self, sys_utterance = None):
        print(f"Got message from dialog system: {sys_utterance}")
        self.socketio.emit('sys_utterance', sys_utterance)
        # socketio.send(sys_utterance)
        # if clients:
        #     socketio.emit('sys_utterance', sys_utterance, roo

if __name__ == "__main__":
    """
    Runs a flask server for connecting the GUI service to the react app.
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret!'
    clients = []
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    service = GUIServer(socketio)

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        clients.append(request.sid)
        socketio.send("Welcome!")

    @socketio.on('disconnect')
    def handle_connect():
        print('Client disconnected')
        clients.remove(request.sid)

    @socketio.on('lang')
    def change_language(language):
        # forward_message_to_react('sys_utterance', language)
        pass

    @socketio.on('start_dialog')
    def start_dialog(message):
        print("SOCKET", message, service)
        return service.start_dialog(message=message)

    @socketio.on('user_utterance')
    def user_utterance(message):
        print(f"SOCKET Message from React: {message}")
        # forward_message_from_react('gen_user_utterance', message)
        return service.user_utterance(message=message)
    
    service.run_standalone()
    socketio.run(app, port=21512)
