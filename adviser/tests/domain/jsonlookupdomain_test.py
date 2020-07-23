import os
import sys
import sqlite3

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
import pytest
from utils.domain.jsonlookupdomain import JSONLookupDomain


@pytest.mark.parametrize('name', ['superhero', 'ImsCourses', 'ImsLecturers'])
def test_load_domain(name):
    """
    Tests the initialization of a JSONLookupDomain with several domain names.

    Args:
        name (str): domain name
    """
    domain = JSONLookupDomain(name)
    assert domain.db is not None
    assert domain.ontology_json is not None
    assert domain.display_name == name


# for the other test, take the domain given in conftest.py
def test_get_root_dir(domain):
    """
    Tests whether retrieving the root directory returns the correct directory.

    Args:
        domain: Domain object (given in conftest.py)
    """
    root_dir = domain._get_root_dir()
    assert root_dir == os.getcwd()  # cwd was set to the root directory at the beginning of this
    # test script


def test_load_db_to_memory(domain, domain_name):
    """
    Tests whether loading the database to memory returns a database connection.

    Args:
        domain: Domain object (given in conftest.py)
        domain_name (str): name of the domain (given in conftest.py)
    """
    db_path = os.path.join('resources', 'databases', domain_name + '.db')
    res = domain._load_db_to_memory(db_path)
    assert res is not None
    assert isinstance(res, sqlite3.Connection)


def test_find_entities_with_constraints(domain, constraint_with_single_match):
    """
    Tests whether querying entities with given constraints returns only entities that match these
    constraints. To make sure about this, a constraint is used in this test that is known to
    match only to a single entity.

    Args:
        domain: Domain object (given in conftest.py)
        constraint_with_single_match (dict): slot-value pair for constraint with a single match
        in the domain (given in conftest_<domain>.py)
    """
    constraints = {constraint_with_single_match['slot']: constraint_with_single_match['value']}
    res = domain.find_entities(constraints=constraints)
    assert len(res) == 1  # because it is a constraint with a single match
    assert domain.get_primary_key() in res[0]


def test_find_entities_without_constraints(domain):
    """
    Tests whether querying entities without specifying any constraints returns several entities.

    Args:
        domain: Domain object (given in conftest.py)
    """
    res = domain.find_entities(constraints={})
    assert len(res) > 0
    assert all(domain.get_primary_key() in entry for entry in res)


def test_find_entities_with_requested_slots(domain, constraintA, constraintB):
    """
    Tests whether querying entities with specifying a list of slots that are requested,
    returns entities
    that contain values for each of these slots.

    Args:
        domain: Domain object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    requested_slots = [constraintA['slot'], constraintB['slot']]
    res = domain.find_entities(constraints={}, requested_slots=requested_slots)
    assert all(constraintA['slot'] in entry and constraintB['slot'] in entry for entry in res)


def test_find_entities_without_requested_slots(domain):
    """
    Tests whether querying entities without specifying any slots returns entry containing the
    primary key and all system requestable slots.

    Args:
        domain: Domain object (given in conftest.py)
    """
    res = domain.find_entities(constraints={}, requested_slots=[])
    slots = {domain.get_primary_key()}
    slots.update(domain.get_system_requestable_slots())
    assert all(set(entry.keys()) == slots for entry in res)


def test_find_info_about_entity_with_requested_slots(domain, constraintA, constraintB, entryA):
    """
    Tests whether querying information for a specific entity with specifying the requestable
    slots returns exactly one entry that contains information for all of these slots (and nothing more).

    Args:
        domain: Domain object (given in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        constraintB (dict): another existing slot-value pair in the domain (given in
        conftest_<domain>.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    prim_key = domain.get_primary_key()
    entity_id = entryA[prim_key]
    requested_slots = [constraintA['slot'], constraintB['slot']]
    res = domain.find_info_about_entity(entity_id, requested_slots)
    assert len(res) == 1
    assert set(res[0].keys()) == set(requested_slots)


def test_find_info_about_entity_without_requested_slots(domain, entryA):
    """
    Tests whether querying information for a specific entity without specifying the requestable
    slots returns exactly one entry with information for several slots.

    Args:
        domain: Domain object (given in conftest.py)
        entryA (dict): slot-value pairs for a complete entry in the domain (given in
        conftest_<domain>.py)
    """
    prim_key = domain.get_primary_key()
    entity_id = entryA[prim_key]
    res = domain.find_info_about_entity(entity_id, requested_slots=[])
    assert len(res) == 1
    assert len(res[0]) > 0


def test_query_db(domain, domain_name):
    """
    Tests whether querying the database returns at least one result with at least one slot.

    Args:
        domain: Domain object (given in conftest.py)
        domain_name (str): name of the domain (given in conftest.py)
    """
    res = domain.query_db('SELECT * FROM {}'.format(domain_name))
    assert len(res) > 0
    assert isinstance(res[0], dict)
    assert len(res[0]) > 0


def test_query_db_without_loaded_db(domain, domain_name):
    """
    Tests whether querying the database without having a database loaded to memory will first
    load the database and then query it.

    Args:
        domain: Domain object (given in conftest.py)
        domain_name (str): name of the domain (given in conftest.py)
    """
    del domain.db
    res = domain.query_db('SELECT * FROM {}'.format(domain_name))
    assert len(res) > 0
    assert isinstance(res[0], dict)
    assert len(res[0]) > 0
    assert 'db' in domain.__dict__
    assert domain.db is not None


def test_get_display_name(domain):
    """
    Tests whether retrieving the display name returns the display name of the domain.

    Args:
        domain: Domain object (given in conftest.py)
    """
    domain.display_name = 'foo'
    display_name = domain.get_display_name()
    assert display_name == domain.display_name


def test_get_requestable_slots(domain, ontology):
    """
    Tests whether retrieving requestable slots returns all requestable slots of the domain.

    Args:
        domain: Domain object (given in conftest.py)
        ontology (dict): Complete ontology of the domain (goven in conftest.py)
    """
    res = domain.get_requestable_slots()
    assert res == ontology['requestable']


def test_get_system_requestable_slots(domain, ontology):
    """
    Tests whether retrieving system requestable slots returns all system requestable slots of the
    domain.

    Args:
        domain: Domain object (given in conftest.py)
        ontology (dict): Complete ontology of the domain (goven in conftest.py)
    """
    res = domain.get_system_requestable_slots()
    assert res == ontology['system_requestable']


def test_get_informable_slots(domain, ontology):
    """
    Tests whether retrieving informable slots returns all informable slots of the domain.

    Args:
        domain: Domain object (given in conftest.py)
        ontology (dict): Complete ontology of the domain (goven in conftest.py)
    """
    res = domain.get_informable_slots()
    assert set(res) == set(ontology['informable'].keys())


def test_get_possible_values(domain, ontology, constraintA):
    """
    Tests whether retrieving possible values for a given slot returns all possible values for
    that slot that are known in the domain.

    Args:
        domain: Domain object (given in conftest.py)
        ontology (dict): Complete ontology of the domain (goven in conftest.py)
        constraintA (dict): an existing slot-value pair in the domain (given in
        conftest_<domain>.py)
    """
    slot = constraintA['slot']
    res = domain.get_possible_values(slot)
    assert res == ontology['informable'][slot]


def test_get_primary_key(domain, ontology):
    """
    Tests whether retrieving the primary key returns the primary key of the domain.

    Args:
        domain: Domain object (given in conftest.py)
        ontology (dict): Complete ontology of the domain (goven in conftest.py)
    """
    res = domain.get_primary_key()
    assert res == ontology['key']


def test_get_pronouns_with_slot_in_pronoun_map(domain, ontology):
    """
    Tests whether retrieving the pronouns for a slot that is listed in the pronoun map of the
    domain returns all pronouns known to this slot.

    Args:
        domain: Domain object (given in conftest.py)
        ontology (dict): Complete ontology of the domain (goven in conftest.py)
    """
    if 'pronoun_map' in ontology:
        slot = ontology['pronoun_map'].keys()[0]
        pronouns = ontology['pronoun_map'][slot]
        res = domain.get_pronouns(slot)
        assert set(res) == set(pronouns)


def test_get_pronouns_without_slot_in_pronoun_map(domain, ontology):
    """
    Tests whether retrieving the pronouns for a slot that is not listed in the pronoun map of the
    domain returns an empty list.

    Args:
        domain: Domain object (given in conftest.py)
        ontology (dict): Complete ontology of the domain (goven in conftest.py)
    """
    if 'pronoun_map' in ontology:
        slots_without_pronouns = [slot for slot in ontology['informable']
                                  if slot not in ontology['pronoun_map']]
        if slots_without_pronouns:
            slot = slots_without_pronouns[0]
            res = domain.get_pronouns(slot)
            assert res == []


def test_get_keyword_with_keyword(domain, ontology):
    """
    Tests whether retrieving the keyword returns the keyword of the domain if there is one.

    Args:
        domain: Domain object (given in conftest.py)
        ontology (dict): Complete ontology of the domain (goven in conftest.py)
    """
    if 'keyword' in ontology:
        res = domain.get_keyword()
        assert res == ontology['keyword']


def test_get_keyword_without_keyword(domain):
    """
    Tests whether retrieving the keyword returns None if there is no keyword in the domain.

    Args:
        domain: Domain object (given in conftest.py)
    """
    del domain.ontology_json['keyword']
    res = domain.get_keyword()
    assert res is None
