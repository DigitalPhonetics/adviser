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

"""Specifies the currently available NLU modules per domain"""

from utils.domain.jsonlookupdomain import JSONLookupDomain
from modules.nlu import HandcraftedNLU

NLU_MODULES = {}

def get_module(domain: JSONLookupDomain, module_name: str):
    return NLU_MODULES[domain][module_name]

def update_modules():
    global NLU_MODULES
    NLU_MODULES = {
        DOMAIN_IMS_COURSES: {
            'Handcrafted NLU for courses at the IMS':
                HandcraftedNLU(DOMAIN_IMS_COURSES)
            },
        DOMAIN_IMS_LECTURERS: {
            'Handcrafted NLU for lecturers at the IMS':
                HandcraftedNLU(DOMAIN_IMS_LECTURERS)
        }
    }


if __name__ == "__main__":
    global DOMAIN_IMS_COURSES, DOMAIN_IMS_LECTURERS

    DOMAIN_IMS_COURSES = JSONLookupDomain("ImsCoursesConfidential")
    DOMAIN_IMS_LECTURERS = JSONLookupDomain("ImsLecturersConfidential")

    # Lists of possible NLU modules for each domain
    
    update_modules()