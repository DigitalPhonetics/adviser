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

from modules.module import Module
from utils import SysAct, SysActionType, UserAct, UserActionType
import re

class MechanicalTurk(Module):
    def __init__(self, domain, logger, user=False):
        super().__init__(domain=domain, logger=logger)
        self.user = user
        self.regex = re.compile(r"(?P<intent>\w+)(?:\((?:(?P<slot>\w+)(?:=(?P<value>\w+))?)?\))?")

    def forward(self, dialog_graph, **kwargs) -> dict(sys_act=SysAct):
        """ Child classes have to overwrite this method """
        
        m_iter = None
        while not m_iter:
            m_iter = self.regex.finditer(input(f"{'System' if not self.user else 'User'} action (intent(slot=value)): "))
            actions = [self.parse_action(**m.groupdict()) for m in m_iter]
            if actions is None:
                # reset match
                m_iter = None
        
        self.logger.dialog_turn(f"{'System' if not self.user else 'User'} Action: {action}")
        return {'sys_act': actions[0]} if not self.user else {'user_acts': actions}

    def parse_action(self, intent, slot, value):
        try:
            if not self.user:
                return SysAct(act_type=SysActionType(intent), slot_values={slot: [value] if value else []} if slot else None)
            else:
                return UserAct(act_type=UserActionType(intent), slot=slot, value=value)
        except ValueError:
            # intent is probably not a valid SysActionType
            if not self.user:
                print("Intent must be one of {}".format(list(map(lambda x: x.value, SysActionType))))
            else:
                print("Intent must be one of {}".format(list(map(lambda x: x.value, UserActionType))))