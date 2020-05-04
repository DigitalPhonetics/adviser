"""
Script for converting a domain's DB file as used for task-oriented dialogue ADVISER to a
knowledge graph file in a Wikidata-like JSON format which can be used by KG management systems
like MongoDB.
This script is meant as a starting point for creating your own domain-specific KG.

Usage of file:
python3 convert_db_to_kg.py YourDomain.db YourDomainKg.json
"""

import sys, os
import json
import sqlite3
from datetime import datetime

def read_db(filename):
    return sqlite3.connect(os.path.join(os.getcwd(), filename)).cursor()

def read_table(cursor, table_name):
    return cursor.execute(f"SELECT * FROM {table_name}")

def get_slot_names(db_query_result):
    return [element[0] for element in db_query_result.description]

def create_properties(slot_names):
    property_list = []
    for i, slot in enumerate(slot_names):
        """property_list.append({
            'description':  '',
            'datatype':     'string',
            'id':           f'PSPEC{i+1}',
            'label':        slot,
            'example':      [],
            'types':        [],
            'aliases':      []
        })"""
        property_list.append({
            'descriptions': {'en': {"language": "en", "value": ''}},
            'title':        f'Property:PSPEC{i+1}',
            'id':           f'PSPEC{i+1}',
            'labels':       {'en': {"language": "en", "value": slot}},
            'datatype':     'string',
            'modified':     datetime.today().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'type':         'property',
            'aliases':      {'en': []},
            'claims':       {}
        })
    return property_list

def create_entities(rows, slots, primary_key_index):
    entity_list = []
    for i, row in enumerate(rows):
        claims = create_claims_for_entity(f'QSPEC{i+1}', slots, row)
        entity_list.append({
            'descriptions': {'en': {"language": "en", "value": ''}},
            'title':        f'QSPEC{i+1}',
            'id':           f'QSPEC{i+1}',
            'labels':       {'en': {"language": "en", "value": row[primary_key_index]}},
            'modified':     datetime.today().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'type':         'item',
            'aliases':      {'en': []},
            'claims':       claims
        })
    return entity_list

def create_claims_for_entity(entity_id, slots, values):
    claims = {}
    for i, (slot, value) in enumerate(zip(slots, values)):
        claims[f'PSPEC{i+1}'] = [{
            'type':         'statement',
            'id':           f'{entity_id}-PSPEC{i+1}-V{value}',
            'rank':         'normal',
            'references':   [],
            'mainsnak':     {
                'snaktype':     'value',
                'property':     f'PSPEC{i+1}',
                'datavalue':    {'value': value, 'type': 'string'},
                'datatype':     'string'
            }
        }]
    return claims

def get_table_name(db):
    result = db.execute("SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';").fetchall()
    assert result, 'No table found in database'
    return result[0][0]

def get_primary_key_index(db, table_name):
    for i, slot in enumerate(db.execute(f'PRAGMA table_info({table_name})').fetchall()):
        if slot[-1] == 1:
            return i
    return None

def write_knowledge_graph(filename, properties, entities):
    ents = {}
    for ent in entities:
        ents[ent['id']] = ent
    for prop in properties:
        ents[prop['id']] = prop
    with open(os.path.join(os.getcwd(), filename), 'w', encoding='utf8') as f:
        json.dump({'entities': ents}, f, indent=4)

if __name__ == '__main__':
    input_file, output_file = sys.argv[1], sys.argv[2]
    db = read_db(input_file)
    table_name = get_table_name(db)
    result = read_table(db, table_name)
    slots = get_slot_names(result)
    rows = result.fetchall()
    properties = create_properties(slots)
    primary_key_index = get_primary_key_index(db, table_name)
    entities = create_entities(rows, slots, primary_key_index)
    write_knowledge_graph(output_file, properties, entities)
