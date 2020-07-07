import os
import sys
import argparse
import pytest


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
from utils.useract import UserActionType, UserAct
from examples.webapi.mensa.domain import MensaDomain
from examples.webapi.mensa.nlu import MensaNLU



def test_setup_of_user_informables_mensa():
    """

    Tests if user informable slots are recognized as such

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)

    for user_informable_slot in ['day', 'type', 'vegan', 'vegetarian', 'fish', 'pork']:
        assert user_informable_slot in nlu.USER_INFORMABLE



def test_setup_of_user_requestables_mensa():
    """

    Tests if user requestable slots are recognized as such

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)
    
    for user_requestable_slot in ['type', 'vegan', 'vegetarian', 'fish', 'pork']:
        assert user_requestable_slot in nlu.USER_REQUESTABLE



# test inform

def test_inform_mensa_day():
    """

    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot-value pair

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "day"
    act_out.value = "today"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='I want information about the mensa plan today')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_inform_mensa_type():
    """

    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot-value pair

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "type"
    act_out.value = "starter"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='I want a starter')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



# test binary inform

def test_inform_binary_mensa_vegan():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "vegan"
    act_out.value = "true"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='vegan, please')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_inform_binary_mensa_vegeterian():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "vegetarian"
    act_out.value = "false"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='not veggi')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out    



def test_inform_binary_mensa_fish():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "fish"
    act_out.value = "true"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='I want it to contain fish')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out  



def test_inform_binary_mensa_pork():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "pork"
    act_out.value = "false"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='I do not want pork')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out  



# test request

def test_request_mensa_type():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "type"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='is this meal a dessert')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_request_mensa_vegan():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "vegan"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='is it vegan')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_request_mensa_vegeterian():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "vegetarian"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='is it veggi')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_request_mensa_fish():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "fish"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='is it fish')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_request_mensa_pork():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "pork"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='does it contain pork')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



# test for multiple user acts

def test_multiple_user_acts_mensa():
    """
    
    Tests exemplary whether a given sentence with multiple user acts is understood properly
    
    """
    domain = MensaDomain()
    nlu = MensaNLU(domain)
    
    usr_utt = nlu.extract_user_acts(nlu, user_utterance='Hi, I want a dish with fish')
    assert 'user_acts' in usr_utt

    act_out = UserAct()
    act_out.type = UserActionType.Hello
    assert usr_utt['user_acts'][0] == act_out

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "fish"
    act_out.value = "true"
    assert usr_utt['user_acts'][1] == act_out