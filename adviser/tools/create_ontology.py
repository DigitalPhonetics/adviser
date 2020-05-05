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

__version__ = "1.0.1"

import argparse
import json
import os
import shutil
import sqlite3
import sys
import importlib

from PyInquirer import prompt, style_from_dict, Token

# taken from the examples package from PyInquirer: https://github.com/CITGuru/PyInquirer/blob/master/examples/__init__.py
# (this way the examples package does not interfere with our examples sub-package)
custom_style_2 = style_from_dict({
    Token.Separator: '#6C6C6C',
    Token.QuestionMark: '#FF9D00 bold',
    #Token.Selected: '',  # default
    Token.Selected: '#5F819D',
    Token.Pointer: '#FF9D00 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#5F819D bold',
    Token.Question: '',
})


class DatabaseTable(object):
    def __init__(self, name):
        self.name = name
        self.fields = []
        self.entries = []

    def _get_slot_id(self, slot):
        for field in self.fields:
            if field[1] == slot:
                return field[0]
        return -1

    def get_slots(self):
        return [field[1] for field in self.fields]

    def get_slot_values(self, slot, dontcare = False):
        # get slot id
        id = self._get_slot_id(slot)
        assert id >= 0, f"Slot '{slot}' is not part of the database table '{self.name}'"
        values = sorted(list(set([entry[id] for entry in self.entries])))
        if dontcare and not ('dontcare' in values or "do n't care" in values):
            values.append('dontcare')
        return  values

class Database(object):
    def __init__(self, path):
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        self.tables = {}

        # result will be (type, name, tbl_name, rootpage, sql)
        cursor.execute("SELECT * FROM sqlite_master where type='table'")

        for _, _, table, _, _ in cursor.fetchall():
            self.tables[table] = DatabaseTable(table)

        for table in self.tables.keys():
            # get fields/slots
            # result will be (id, name, type, not null, default, primary key)
            cursor.execute(f"PRAGMA table_info({table});")
            self.tables[table].fields = cursor.fetchall()
            # make sure that fields are sorted according to field index (should be already anyway)
            self.tables[table].fields = sorted(self.tables[table].fields, key=lambda field: field[0])

            # get entries (especially for possible values) 
            cursor.execute(f"SELECT * FROM {table}")
            self.tables[table].entries = cursor.fetchall()

            # add user and system actions
        
    def get_tables(self):
        return list(self.tables.keys())

    def get_slots(self, table):
        return self.tables[table].get_slots()

    def get_slot_values(self, table, slot):
        return self.tables[table].get_slot_values(slot)

def get_defaults():
    return {'discourseAct': ["ack", "hello", "none", "silence", "thanks", "bad"],
            'method': ["none", "byconstraints", "byprimarykey", "finished", "byalternatives", "restart"],
            'key': 'name'}

def run_questions(db: Database):
    # initialize with default values
    answers = get_defaults()

    questions = [
        {
            'type': 'list',
            'qmark': '>>>',
            'message': 'Select table to create ontology for',
            'name': 'table',
            'choices': [{'key': str(id), 'name': table, 'value': table} for id, table in enumerate(db.get_tables())],
            'validate': lambda answer: 'You must choose at least one table.' \
                if len(answer) == 0 else True
        },
        {
            'type': 'input',
            'qmark': '>>>',
            'message': 'Enter the name of the domain:',
            'name': 'domain',
            'default': lambda answers: answers['table']
        },
        {
            'type': 'list',
            'qmark': '>>>',
            'name': 'key',
            'message': 'Which slot will be used as key? (The key uniquely identifies an entity in the database, e.g. the name in case of restaurants)',
            'choices': lambda answers: [{'name': slot} for slot in db.get_slots(answers['table'])]
        },
        {
            'type': 'checkbox',
            'qmark': '>>>',
            'name': 'requestable',
            'message': 'Select user requestables',
            'choices': lambda answers: [{'name': slot, 'checked': slot != 'id'} for slot in db.get_slots(answers['table'])]
        },
        {
            'type': 'checkbox',
            'qmark': '>>>',
            'name': 'system_requestable',
            'message': 'Select system requestables',
            'choices': lambda answers: [{'name': slot} for slot in db.get_slots(answers['table'])]
        },
        {
            'type': 'checkbox',
            'qmark': '>>>',
            'name': 'informable',
            'message': 'Select informable slots',
            'choices': lambda answers: [{'name': slot} for slot in db.get_slots(answers['table'])]
        }]
    answers_ = prompt(questions, style=custom_style_2)
    # check whether there are answers (e.g. if the user cancels the prompt using Ctrl+c)
    if not answers_:
        exit()
    answers.update(answers_)
    
    # get values for informable slots
    questions = [
        {
            'type': 'checkbox',
            'qmark': '>>>',
            'name': slot,
            'message': f'Select values for informable slot {slot}',
            'choices': [{'name': value, 'checked': value != 'dontcare'} for value in db.get_slot_values(answers['table'], slot)]
        } for slot in answers['informable']
    ]

    values = prompt(questions, style=custom_style_2)
    # merge informable slot values with informable slots
    answers['informable'] = {slot: values[slot] for slot in answers['informable'] if slot in values}

    # get binary slots
    questions = [
        {
            'type': 'checkbox',
            'qmark': '>>>',
            'name': 'binary',
            'message': 'Select binary slots',
            'choices': [{'name': slot, 'checked': set(db.get_slot_values(answers['table'], slot)) == {'true', 'false'}} for slot in list(answers['informable'].keys()) + answers['requestable'] + answers['system_requestable']]
        }
    ]
    answers_ = prompt(questions, style=custom_style_2)
    # check whether there are answers (e.g. if the user cancels the prompt using Ctrl+c)
    if not answers_:
        exit()
    answers.update(answers_)
    

    return answers

if __name__ == "__main__":
    sys.tracebacklimit = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='path to the database from which the ontology will be created')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    args = parser.parse_args()

    if not os.path.isfile(args.database):
        print("The supplied database does not exist. Please make sure to provide a proper path.")
        exit(1)
    
    print(f"\n\033[1;37mWelcome to the ontology creation tool v{__version__}!\n\nThis tool will help you to create an ontology from a database.\nSelected options are marked with a filled blue circle.\n\033[0;0m")
    
    # create database object and extract information
    db = Database(args.database)

    if not db.get_tables():
        # there are no tables in the database
        print("Error: The given database is empty. Please add at least one table.")
        exit(1)

    ask_questions = True
    while ask_questions:
        # ask questions
        answers = run_questions(db)
        print("The ontology looks like the following:")
        ont = {key: answers[key] for key in filter(lambda key: key not in ['table', 'domain', 'discourseAct', 'method'], answers.keys())} # exclude some keys
        print(json.dumps(ont, indent=4, sort_keys=True))
        questions = [{
            'type': 'confirm',
            'qmark': '>>>',
            'message': 'Do you want to change anything?',
            'name': 'ask_questions',
            'default': False,
        }]
        ask_questions = prompt(questions, style=custom_style_2)
        # check whether there are answers (e.g. if the user cancels the prompt using Ctrl+c)
        if not ask_questions:
            exit()
        ask_questions = ask_questions['ask_questions']

    # get domain and remove unnecessary elements from the dict
    domain = answers['domain']
    del answers['domain'], answers['table']
    # remove binary list if empty
    if not answers['binary']:
        del answers['binary']

    # ask user about filename
    filename_ont = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../resources/ontologies/{domain}.json'))
    filename_db = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../resources/databases/{domain}.db'))
    questions = [
        {
            'type': 'input',
            'qmark': '>>>',
            'message': 'Enter the path including the filename where the ontology will be stored:',
            'name': 'filename_ont',
            'default': filename_ont
        },
        {
            'type': 'confirm',
            'qmark': '>>>',
            'message': 'Do you want to save the database?',
            'name': 'save_db',
            'default': True,
        },
        {
            'type': 'input',
            'qmark': '>>>',
            'message': 'Enter the path including the filename where the database will be stored:',
            'name': 'filename_db',
            'default': filename_db,
            'when': lambda answers: answers.get('save_db', True)
        }
    ]
    filenames = prompt(questions, style=custom_style_2)

    # check whether there are answers (e.g. if the user cancels the prompt using Ctrl+c)
    if not filenames:
        exit()
    
    # save ontology
    with open(filenames['filename_ont'], 'w') as fp:
        json.dump(answers, fp, indent=4, sort_keys=True)

    # save database
    if filenames['save_db']:
        shutil.copy2(args.database, filenames['filename_db'])