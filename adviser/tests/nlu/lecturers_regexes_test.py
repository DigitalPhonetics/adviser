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



def test_setup_of_user_informables_lecturers():
    """
    
    Tests if user informable slots are recognized as such

    """
    domain = JSONLookupDomain('ImsLecturers')
    nlu = HandcraftedNLU(domain)
    
    for user_informable_slot in ['name', 'department', 'position']:
        assert user_informable_slot in nlu.USER_INFORMABLE



def test_setup_of_user_requestables_lecturers():
    """

    Tests if user requestable slots are recognized as such

    """
    domain = JSONLookupDomain('ImsLecturers')
    nlu = HandcraftedNLU(domain)
    
    for user_requestable_slot in ['name', 'department', 'office_hours', 'mail', 'phone', 'room', 'position']:
        assert user_requestable_slot in nlu.USER_REQUESTABLE



# test inform

def test_inform_lecturers_department():
    """

    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot-value pair

    """
    domain = JSONLookupDomain('ImsLecturers')
    nlu = HandcraftedNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "department"
    act_out.value = "external"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='informatics')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_inform_lecturers_name():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot-value pair

    """
    domain = JSONLookupDomain('ImsLecturers')
    nlu = HandcraftedNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "name"
    act_out.value = "apl. prof. dr. agatha christie"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='agatha christie')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



# test request

def test_request_lecturers_phone():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

    """
    domain = JSONLookupDomain('ImsLecturers')
    nlu = HandcraftedNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "phone"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='can you tell me the phone number')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



def test_request_lecturers_office_hours():
    """
    
    Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

    """
    domain = JSONLookupDomain('ImsLecturers')
    nlu = HandcraftedNLU(domain)
   
    act_out = UserAct()
    act_out.type = UserActionType.Request
    act_out.slot = "office_hours"

    usr_utt = nlu.extract_user_acts(nlu, user_utterance='when are the consultation hours')
    assert 'user_acts' in usr_utt
    assert usr_utt['user_acts'][0] == act_out



# test for multiple user acts

def test_multiple_user_acts_lecturers():
    """
    
    Tests exemplary whether a given sentence with multiple user acts is understood properly
    
    """
    domain = JSONLookupDomain('ImsLecturers')
    nlu = HandcraftedNLU(domain)
    
    usr_utt = nlu.extract_user_acts(nlu, user_utterance='Hi, I want a lecturer who is responsible for gender issues')
    assert 'user_acts' in usr_utt

    act_out = UserAct()
    act_out.type = UserActionType.Hello
    assert usr_utt['user_acts'][0] == act_out

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "position"
    act_out.value = "gender"
    assert usr_utt['user_acts'][1] == act_out