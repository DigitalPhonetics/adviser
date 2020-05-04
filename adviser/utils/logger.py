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

""" This module provides a logger for configurable output on different levels. """
import datetime
import logging
import os
import sys
from enum import IntEnum


class LogLevel(IntEnum):
    """ The available levels for the logger. """
    DIALOGS = 18
    RESULTS = 19
    INFO = 20
    ERRORS = 40
    NONE = 100


# register log levels
logging.addLevelName(int(LogLevel.DIALOGS), 'DIALOGS')
logging.addLevelName(int(LogLevel.RESULTS), 'RESULTS')
logging.addLevelName(int(LogLevel.NONE), 'NONE')


def exception_logging_hook(exc_type, exc_value, exc_traceback):
    """ Used as a hook to log exceptions. """
    logging.getLogger('adviser').error("Uncaught exception",
                                       exc_info=(exc_type, exc_value, exc_traceback))


class MultilineFormatter(logging.Formatter):
    """ A formatter for the logger taking care of multiline messages. """

    def format(self, record: logging.LogRecord):
        save_msg = record.msg
        output = ""
        for idx, line in enumerate(save_msg.splitlines()):
            if idx > 0:
                output += "\n"
            record.msg = line
            output += super().format(record)
        record.msg = save_msg
        record.message = output
        return output


class DiasysLogger(logging.Logger):
    """Logger class.

    This class enables logging to both a logfile and the console with different
    information levels.
    It also provides logging methods for the newly introduced information
    levels (LogLevel.DIALOGS and LogLevel.RESULTS).

    If file_level is set to LogLevel.NONE, no log file will be created.
    Otherwise, the output directory can be configured by setting log_folder.

    """

    def __init__(self, name: str = 'adviser', console_log_lvl: LogLevel = LogLevel.ERRORS,
                 file_log_lvl: LogLevel = LogLevel.NONE, logfile_folder: str = 'logs',
                 logfile_basename: str = 'log'):  # pylint: disable=too-many-arguments
        super(DiasysLogger, self).__init__(name)

        if file_log_lvl is not LogLevel.NONE:
            # configure output to log file
            os.makedirs(os.path.realpath(logfile_folder), exist_ok=True)
            log_file_name = logfile_basename + '_' + \
                            str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.log'
            log_file_path = os.path.join(os.path.realpath(logfile_folder), log_file_name)

            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setLevel(int(file_log_lvl))

            fh_formatter = MultilineFormatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(fh_formatter)
            self.addHandler(file_handler)

        # configure output to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(int(console_log_lvl))
        # ch_formatter = MultilineFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch_formatter = MultilineFormatter('logger: %(message)s')
        console_handler.setFormatter(ch_formatter)
        self.addHandler(console_handler)

        # log exceptions
        sys.excepthook = exception_logging_hook

    def result(self, msg: str):
        """ Logs the result of a dialog """

        self.log(int(LogLevel.RESULTS), msg)

    def dialog_turn(self, msg: str, dialog_act=None):
        """ Logs a turn of a dialog """

        log_msg = msg
        if dialog_act is not None:
            log_msg += "\n  " + str(dialog_act)
        self.log(int(LogLevel.DIALOGS), log_msg)
