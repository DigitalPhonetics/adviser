import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())

# Parameters for the domain "superhero" that can be used in the tests. A constraint is a
# slot-value pair that should exist in this domain.


@pytest.fixture
def domain_name():
    """
    The name of the domain.
    """
    return 'superhero'


@pytest.fixture
def primkey_constraint():
    """
    Constraint having the primary key of the domain as slot.
    """
    return {'slot': 'name', 'value': 'Batman'}


@pytest.fixture
def constraintA():
    """
    Constraint for a system requestable slot.
    """
    return {'slot': 'main_superpower', 'value': 'Magic'}


@pytest.fixture
def constraintA_alt():
    """
    Constraint with the same slot as constraintA, but a different value.
    """
    return {'slot': 'main_superpower', 'value': 'Claws'}


@pytest.fixture
def constraintB():
    """
    Constraint for a system requestable slot that is different to the slot in constraintA.
    """
    return {'slot': 'primary_uniform_color', 'value': 'Black'}


@pytest.fixture
def constraint_with_multiple_matches():
    """
    Constraint for a system requestable slot. This constraint matches multiple entries in the
    database.
    """
    return {'slot': 'main_superpower', 'value': 'Magic'}


@pytest.fixture
def constraint_with_single_match():
    """
    Constraint for a system requestable slot. This constraint matches exactly one entry in the
    database.
    """
    return {'slot': 'main_superpower', 'value': 'Claws'}


@pytest.fixture
def constraint_with_no_match():
    """
    Constraint for a system requestable slot. This constraint matches no entry in the database
    (i.e. the value is not one of the possible values for the slot).
    """
    return {'slot': 'main_superpower', 'value': 'Fire'}


@pytest.fixture
def constraint_binaryA():
    """
    Constraint for a binary, system requestable slot. Binary slots have only two possible values
    (e.g. true and false). The constraint is empty here because this domain does not contain
    binary slots.
    """
    return {}


@pytest.fixture
def constraint_binaryB():
    """
    Constraint for a binary, system requestable slot that is different to the slot in
    constraint_binaryA. Binary slots have only two  possible values (e.g. true and false). The
    constraint is empty here because this domain does not contain binary slots.
    """
    return {}


@pytest.fixture
def entryA():
    """
    One complete entry of the database.
    """
    return {
        'name': 'Batman',
        'primary_uniform_color': 'Black',
        'main_superpower': 'Gadgets',
        'last_known_location': 'Gotham City',
        'loyalty': 'He works best alone',
        'description': 'Some description.',
        'real_name': 'Bruce Wane'}


@pytest.fixture
def entryB():
    """
    One complete entry of the database that is different to entryA.
    """
    return {
        'name': 'Superman',
        'primary_uniform_color': 'Blue',
        'main_superpower': 'Strength',
        'last_known_location': 'Metropolis City',
        'loyalty': 'Justice League',
        'description': 'Some description.',
        'real_name': 'Clark Kent'}
