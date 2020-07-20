import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())

# Parameters for the domain "ImsLecturers" that can be used in the tests. A constraint is a
# slot-value pair that should exist in this domain.


@pytest.fixture
def domain_name():
    """
    The name of the domain.
    """
    return 'ImsLecturers'


@pytest.fixture
def primkey_constraint():
    """
    Constraint having the primary key of the domain as slot.
    """
    return {'slot': 'name', 'value': 'george orwell'}


@pytest.fixture
def constraintA():
    """
    Constraint for a system requestable slot.
    """
    return {'slot': 'position', 'value': 'professor'}


@pytest.fixture
def constraintA_alt():
    """
    Constraint with the same slot as constraintA, but a different value.
    """
    return {'slot': 'position', 'value': 'adviser'}


@pytest.fixture
def constraintB():
    """
    Constraint for a system requestable slot that is different to the slot in constraintA.
    """
    return {'slot': 'department', 'value': 'foundations'}


@pytest.fixture
def constraint_with_multiple_matches():
    """
    Constraint for a system requestable slot. This constraint matches multiple entries in the
    database.
    """
    return {'slot': 'position', 'value': 'professor'}


@pytest.fixture
def constraint_with_single_match():
    """
    Constraint for a system requestable slot. This constraint matches exactly one entry in the
    database.
    """
    return {'slot': 'position', 'value': 'adviser'}


@pytest.fixture
def constraint_with_no_match():
    """
    Constraint for a system requestable slot. This constraint matches no entry in the database
    (i.e. the value is not one of the possible values for the slot).
    """
    return {'slot': 'position', 'value': 'teacher'}


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
        'name': 'george orwell',
        'department': 'phonetics',
        'office_hours': 'Thu., 15:00-16:00',
        'mail': 'contact@ims.uni-stuttgart.de',
        'phone': '0711/123456',
        'room': 'PWR 05B – 02.010',
        'cap_name': 'George Orwell',
        'gender': 'male'}


@pytest.fixture
def entryB():
    """
    One complete entry of the database that is different to entryA.
    """
    return {
        'name': 'dr. emily dickinson',
        'department': 'na',
        'office_hours': 'by appointment',
        'mail': 'contact@ims.uni-stuttgart.de',
        'phone': '0711/123456',
        'room': 'PWR 05B – 01.006',
        'cap_name': 'Dr. Emily Dickinson',
        'gender': 'female'}

