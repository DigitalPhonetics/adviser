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

from services.service import RemoteService

import webbrowser
import os

class Webui(RemoteService):
    def __init__(self, identifier="webui"):
        super().__init__(identifier=identifier)
        # open UI in webbrowser automatically
        webui_path = f"file:///{os.path.join(os.path.realpath(''), 'tools', 'webui', 'chat.html')}"
        print("WEBUI accessible at", webui_path)
        # TODO re-enable webbrowser.open(webui_path) 
