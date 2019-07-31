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
from utils import SysAct, SysActionType
import re

class MechanicalTurk(Module):

    def forward(self, dialog_graph, **kwargs) -> dict(sys_act=SysAct):
        """ Child classes have to overwrite this method """
        
        p = re.compile(r"^(?P<intent>\w+)(\(((?P<slot>\w+)(=(?P<value>\w+))?)?\))?$")
        m = None
        while not m:
            m = p.match(input("System action (intent(slot=value)): "))
            sys_action = self.parse_action(**m.groupdict())
            if sys_action is None:
                # reset match
                m = None

        return {'sys_act': sys_action}

    def parse_action(self, intent, slot, value):
        try:
            return SysAct(act_type=SysActionType(intent), slot_values={slot: [value] if value else []} if slot else None)
        except ValueError:
            # intent is probably not a valid SysActionType
            print("Intent must be one of {}".format(list(map(lambda x: x.value, SysActionType))))
