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

import sys
from PyQt5.QtWidgets import QApplication
from tools.crosstickgame.startscreen import StartScreen

from utils import UserAct, UserActionType

import re


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StartScreen()
    for arg in sys.argv[1:]:
        window.add(arg)
    window.show()
    app.exec_()
