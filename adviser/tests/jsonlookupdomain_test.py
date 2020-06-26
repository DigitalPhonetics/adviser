import os
import sys
import argparse

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
import pytest
from utils.domain.jsonlookupdomain import JSONLookupDomain


# Test case using pytest.fixture

@pytest.fixture(scope="module")
def load_superhero_domain():
    """ Try loading the restaurant domain shared by all following tests in this file. """
    domain = JSONLookupDomain('superhero')

    assert domain.db is not None
    assert domain.ontology_json is not None

    return domain


def test_query_db(load_supehero_domain : JSONLookupDomain):
    """
        Test functionality: query_db executes SQL SELECT query
        
        Note, a better test would have a fixed testing db rather than one which could change
    """
    query_str = "SELECT * FROM supehero"
    domain = load_superhero_domain
    entities = domain.query_db(query_str)

    assert len(entities) > 0   # make sure we found some data
    # print(len(entities))

    assert isinstance(entities[0], dict) # make sure items are dictionaries
    assert len(entities[0]) > 0          # we want all columns from the database
    # print(entities[0])


