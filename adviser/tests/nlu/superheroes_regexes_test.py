import os
import sys
import argparse
import pytest


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
from utils.useract import UserActionType, UserAct
from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.nlu.nlu import HandcraftedNLU



def test_setup_of_user_informables_lecturers(nlu):
    """
    
    Tests exemplary whether a given user utterance, is recognized as belonging to a certain slot-value pair

    Args: 
        nlu: NLU Object (given in conftest.py)

    """
    for user_informable_slot in ['loyalty', 'main_superpower', 'name', 'primary_uniform_color']:
        assert user_informable_slot in nlu.USER_INFORMABLE



def test_setup_of_user_requestables_lecturers(nlu):
    """

    Tests exemplary whether a given user utterance, is recognized as belonging to a certain slot-value pair

    Args: 
        nlu: NLU Object (given in conftest.py)

    """
    for user_requestable_slot in ['loyalty', 'main_superpower', 'name', 'primary_uniform_color', 'last_known_location', 'description', 'real_name']:
        assert user_requestable_slot in nlu.USER_REQUESTABLE



# test inform

def test_inform_superheroes_primary_uniform_color(nlu):
    """

    Tests exemplary whether a given user utterance, is recognized as belonging to a certain slot-value pair

    Args: 
        nlu: NLU Object (given in conftest.py)

    """
    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "primary_uniform_color"
    act_out.value = "Purple"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='the uniform should be Purple')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_inform_superheroes_loyalty(nlu):
    """
    
    Tests exemplary whether a given user utterance, is recognized as belonging to a certain slot-value pair

    Args: 
        nlu: NLU Object (given in conftest.py)

    """
    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "loyalty"
    act_out.value = "Avengers"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='they should be part of Avengers')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



# test request

def test_request_superheroes_last_known_location(nlu):
    """
    
    Tests exemplary whether a given user utterance, is recognized as belonging to a certain slot

    Args: 
        nlu: NLU Object (given in conftest.py)

    """
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "last_known_location"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='what is their location')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_request_superheroes_real_name(nlu):
    """
    
    Tests exemplary whether a given user utterance, is recognized as belonging to a certain slot

    Args: 
        nlu: NLU Object (given in conftest.py)

    """
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "real_name"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='what is their real name')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out