############################################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify'
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
############################################################################################


class Topic(object):
    DIALOG_START = 'dialog_start'  # Called at the beginning of a new dialog. Subscribe here to set stateful variables for one dialog.
    DIALOG_END = 'dialog_end'      # Called at the end of a dialog (after a bye-action).
    DIALOG_EXIT = 'dialog_exit'    # Called when the dialog system shuts down. Subscribe here if you e.g. have to close resource handles / free locks.
