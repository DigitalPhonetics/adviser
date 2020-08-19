import os
import sys
import json
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
from utils.beliefstate import BeliefState
from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.policy import HandcraftedPolicy
from services.bst import HandcraftedBST
from services.simulator.goal import Goal
from services.nlu.nlu import HandcraftedNLU
from services.nlg import HandcraftedNLG
from services.nlg.affective_nlg import HandcraftedEmotionNLG

from services.simulator.simulator import HandcraftedUserSimulator, Agenda


pytest_plugins = ['conftest_superhero', 'conftest_services']

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "terminate_topics: mark fixture to use a specific list of terminate topics"
    )


@pytest.fixture
def domain(domain_name):
    return JSONLookupDomain(domain_name)


@pytest.fixture
def ontology(domain_name):
    return json.load(open(os.path.join(get_root_dir(), 'resources', 'ontologies',
                                       domain_name + '.json')))


@pytest.fixture
def beliefstate(domain):
    bs = BeliefState(domain)
    return bs

@pytest.fixture
def policy(domain):
    policy = HandcraftedPolicy(domain)
    policy.dialog_start()
    return policy

@pytest.fixture
def bst(domain):
    return HandcraftedBST(domain)

@pytest.fixture
def goal(domain):
    return Goal(domain)

@pytest.fixture
def nlu(domain):
	nlu = HandcraftedNLU(domain)
	return nlu

@pytest.fixture
def nlg(domain):
    nlg = HandcraftedNLG(domain)
    return nlg

@pytest.fixture
def aff_nlg(domain):
    aff_nlg = HandcraftedEmotionNLG(domain)
    return aff_nlg

def agenda():
    return Agenda()

@pytest.fixture
def simulator(domain):
    simulator = HandcraftedUserSimulator(domain)
    simulator.dialog_start()
    return simulator
