import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())


@pytest.fixture
def domain_name():
    return 'superhero'


@pytest.fixture
def primkey_constraint():
    return {'slot': 'name', 'value': 'Batman'}


@pytest.fixture
def constraintA():
    return {'slot': 'main_superpower', 'value': 'Magic'}


@pytest.fixture
def constraintB():
    return {'slot': 'primary_uniform_color', 'value': 'Black'}


@pytest.fixture
def constraint_with_multiple_matches():
    return {'slot': 'main_superpower', 'value': 'Magic'}


@pytest.fixture
def constraint_with_single_match():
    return {'slot': 'main_superpower', 'value': 'Claws'}


@pytest.fixture
def constraint_with_no_match():
    return {'slot': 'main_superpower', 'value': 'Fire'}