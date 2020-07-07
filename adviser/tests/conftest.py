import os
import sys
import argparse
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
from services.nlg.nlg import HandcraftedNLG



pytest_plugins = ['conftest_superhero']


@pytest.fixture
def domain(domain_name):
    return JSONLookupDomain(domain_name)

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