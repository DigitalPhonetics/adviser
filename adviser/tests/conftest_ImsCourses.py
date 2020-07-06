import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())

# Parameters for the domain "ImsCourses" that can be used in the tests. A constraint is a
# slot-value pair that should exist in this domain.


@pytest.fixture
def domain_name():
    """
    The name of the domain.
    """
    return 'ImsCourses'


@pytest.fixture
def primkey_constraint():
    """
    Constraint having the primary key of the domain as slot.
    """
    return {'slot': 'name', 'value': 'parsing'}


@pytest.fixture
def constraintA():
    """
    Constraint for a system requestable slot.
    """
    return {'slot': 'ects', 'value': '3'}


@pytest.fixture
def constraintA_alt():
    """
    Constraint with the same slot as constraintA, but a different value.
    """
    return {'slot': 'ects', 'value': '6'}


@pytest.fixture
def constraintB():
    """
    Constraint for a system requestable slot that is different to the slot in constraintA.
    """
    return {'slot': 'course_type', 'value': 'vu'}


@pytest.fixture
def constraint_with_multiple_matches():
    """
    Constraint for a system requestable slot. This constraint matches multiple entries in the
    database.
    """
    return {'slot': 'ects', 'value': '3'}


@pytest.fixture
def constraint_with_single_match():
    """
    Constraint for a system requestable slot. This constraint matches exactly one entry in the
    database.
    """
    return {'slot': 'ects', 'value': '9'}


@pytest.fixture
def constraint_with_no_match():
    """
    Constraint for a system requestable slot. This constraint matches no entry in the database
    (i.e. the value is not one of the possible values for the slot).
    """
    return {'slot': 'ects', 'value': '15'}


@pytest.fixture
def constraint_binaryA():
    """
    Constraint for a binary, system requestable slot. Binary slots have only two possible values
    (e.g. true and false)
    """
    return {'slot': 'bachelor', 'value': 'true'}


@pytest.fixture
def constraint_binaryB():
    """
    Constraint for a binary, system requestable slot that is different to the slot in
    constraint_binaryA. Binary slots have only two  possible values (e.g. true and false).
    """
    return {'slot': 'speech', 'value': 'false'}


@pytest.fixture
def entryA():
    """
    One complete entry of the database.
    """
    return {
        'name': 'parsing',
        'id': '401516004',
        'ects': '6',
        'turn': 'sose',
        'sws': '4',
        'lang': 'en',
        'lecturer': 'apl. prof. dr. agatha christie',
        'module_id': '13270',
        'module_name': 'parsing',
        'course_type': 'vu',
        'time_slot': 'thu., 09:45-11:15 and fri., 09:45-11:15',
        'description': 'na',
        'bachelor': 'true',
        'master': 'true',
        'elective': 'true',
        'room': 'v 5.01 (pfaffenwaldring 5b)',
        'requirements': 'na',
        'examination_style': 'written exam, project, and project report',
        'project': 'true',
        'written_exam': 'true',
        'oral_exam': 'false',
        'presentation': 'false',
        'report': 'true',
        'participation_limit': 'no',
        'obligatory_attendance': 'false',
        'speech': 'false',
        'linguistics': 'true',
        'applied_nlp': 'true',
        'machine_learning': 'false',
        'deep_learning': 'false',
        'cognitive_science': 'false',
        'semantics': 'false',
        'syntax': 'true',
        'statistics': 'false',
        'programming': 'false',
        'extendable': 'false',
        'cap_lecturer': 'Apl. Prof. Dr. Agatha Christie'
    }


@pytest.fixture
def entryB():
    """
    One complete entry of the database that is different to entryA.
    """
    return {
        'name': 'dialog modeling',
        'id': '401400040',
        'ects': '3',
        'turn': 'wise',
        'sws': '2',
        'lang': 'en',
        'lecturer': 'george orwell',
        'module_id': '75690',
        'module_name': 'spoken language understanding',
        'course_type': 'se',
        'time_slot': 'wed., 15:45-17:15',
        'description': 'some description',
        'bachelor': 'false',
        'master': 'true',
        'elective': 'true',
        'room': 'v 5.01 (pfaffenwaldring 5b)',
        'requirements': 'some requirements',
        'examination_style': 'presentation and report',
        'project': 'false',
        'written_exam': 'false',
        'oral_exam': 'false',
        'presentation': 'true',
        'report': 'true',
        'participation_limit': 'yes',
        'obligatory_attendance': 'true',
        'speech': 'true',
        'linguistics': 'false',
        'applied_nlp': 'true',
        'machine_learning': 'true',
        'deep_learning': 'false',
        'cognitive_science': 'false',
        'semantics': 'false',
        'syntax': 'false',
        'statistics': 'false',
        'programming': 'false',
        'extendable': 'false',
        'cap_lecturer': 'George Orwell'}

