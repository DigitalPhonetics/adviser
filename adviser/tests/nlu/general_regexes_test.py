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



# Tests for general acts

def test_hello(nlu):
    """

    Tests whether a given user input is identified as general act: Hello

    Args:
        nlu: NlU Object (given in conftest.py)

    """
    for input in ["hi", "hello", "howdy", "hey"]:
        act_out = UserAct()
        act_out.type = UserActionType.Hello

        usr_utt = nlu.extract_user_acts(nlu, user_utterance=input)

        assert 'user_acts' in usr_utt
        assert usr_utt['user_acts'][0] == act_out



def test_bye(nlu):
    """

    Tests whether a given user input is identified as general act: Bye

    Args:
        nlu: NLU Object (given in conftest.py)

    """
    for input in ["bye", "goodbye", "that is all", "that's all"]:
        act_out = UserAct()
        act_out.type = UserActionType.Bye

        usr_utt = nlu.extract_user_acts(nlu, user_utterance=input)
    
        assert 'user_acts' in usr_utt
        assert usr_utt['user_acts'][0] == act_out



def test_deny(nlu):
    """

    Tests whether a given user input is identified as general act: Deny

    Args:
        nlu: NLU Object (given in conftest.py)

    """
    for input in ["no", "not true", "wrong", "error", "n", "nope", "incorrect", "not correct"]:
        act_out = UserAct()
        act_out.type = UserActionType.Deny

        usr_utt = nlu.extract_user_acts(nlu, user_utterance=input) 
        
        assert 'user_acts' in usr_utt
        assert usr_utt['user_acts'][0] == act_out



def test_affirm(nlu):
    """

    Tests whether a given user input is identified as general act: Affirm

    Args:
        nlu: NLU Object (given in conftest.py)

    """
    for input in ["yes", "Yeah", "Ok", "Sure", "right", "correct"]:
        act_out = UserAct()
        act_out.type = UserActionType.Affirm

        usr_utt = nlu.extract_user_acts(nlu, user_utterance=input) 

        assert 'user_acts' in usr_utt
        assert usr_utt['user_acts'][0] == act_out



def test_thanks(nlu):
    """

    Tests whether a given user input is identified as general act: Thanks

    Args:
        nlu: NLU Object (given in conftest.py)

    """
    for input in ["Great", "Thanks", "thank you", "awesome", "thank you so much"]:
        act_out = UserAct()
        act_out.type = UserActionType.Thanks

        usr_utt = nlu.extract_user_acts(nlu, user_utterance=input) 

        assert 'user_acts' in usr_utt
        assert usr_utt['user_acts'][0] == act_out



def test_reqalts(nlu):
    """

    Tests whether a given user input is identified as general act: RequestAlternatives

    Args:
        nlu: NLU Object (given in conftest.py)

    """
    for input in ["something else", "anything else", "different one", "another one", "don't want that", "other options"]:
        act_out = UserAct()
        act_out.type = UserActionType.RequestAlternatives

        usr_utt = nlu.extract_user_acts(nlu, user_utterance='something else') 

        assert 'user_acts' in usr_utt
        assert usr_utt['user_acts'][0] == act_out