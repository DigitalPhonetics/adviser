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

import argparse
import os
import sys

head_location = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..')) # main folder of adviser
sys.path.append(head_location)

import json
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.useract import UserAct, UserActionType
from tools.regextemplates.rules.regexfile import RegexFile


def _write_dict_to_file(dict_object: dict, filename: str):
    file_path = os.path.join(head_location, 'resources', 'nlu_regexes', filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dict_object, file, sort_keys=True)


def _create_request_json(domain: JSONLookupDomain, template: RegexFile):
    request_regex_json = {}
    for slot in domain.get_requestable_slots():
        request_act = UserAct(act_type=UserActionType.Request, slot=slot)
        request_regex_json[slot] = template.create_regex(request_act)
    return request_regex_json


def _create_inform_json(domain: JSONLookupDomain, template: RegexFile):
    inform_regex_json = {}
    for slot in domain.get_informable_slots():
        inform_regex_json[slot] = {}
        for value in domain.get_possible_values(slot):
            inform_act = UserAct(act_type=UserActionType.Inform, slot=slot, value=value)
            inform_regex_json[slot][value] = template.create_regex(inform_act)
    return inform_regex_json


def create_json_from_template(domain: JSONLookupDomain, template_filename: str):
    template = RegexFile(template_filename, domain)
    domain_name = domain.get_domain_name()
    _write_dict_to_file(_create_request_json(domain, template), f'{domain_name}RequestRules.json')
    _write_dict_to_file(_create_inform_json(domain, template), f'{domain_name}InformRules.json')


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", help="name of the domain")
    parser.add_argument("filename", help="name of your .nlu file without the .nlu ending (e.g.: resources/nlu_regexes/YOURNLUFILE.nlu -> provide YOURFILE)")
    args = parser.parse_args()
    nlu_file = os.path.join(head_location, 'resources', 'nlu_regexes', f"{args.filename}.nlu")
    dom = JSONLookupDomain(args.domain)
    create_json_from_template(dom, nlu_file)
