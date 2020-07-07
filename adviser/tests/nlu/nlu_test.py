import os
import sys
import argparse
import pytest
import re


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
from utils.useract import UserActionType, UserAct
from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.nlu.nlu import HandcraftedNLU
from utils.common import Language
from utils.sysact import SysAct, SysActionType



def test_init(nlu, domain):
	"""

	Tests whether given class is initialized properly

	Args:
		nlu: NLU Object (given in conftest.py)
		domain: Domain Object (given in conftest.py)

	"""
	nlu.__init__(domain)

	assert nlu.language == Language.ENGLISH
	assert nlu.domain_name == 'superhero'
	assert nlu.domain_key == 'name'
	assert nlu.sys_act_info == {'last_act': None, 'lastInformedPrimKeyVal': None, 'lastRequestSlot': None}
	for i in ['loyalty', 'main_superpower', 'name', 'primary_uniform_color']:
		assert i in nlu.USER_INFORMABLE
	for i in ['loyalty', 'main_superpower', 'name', 'primary_uniform_color', 'last_known_location', 'description', 'real_name']:
		assert i in nlu.USER_REQUESTABLE



def test_match_general_act_hello(nlu):
	"""

	Tests if general act hello are matched properly

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	nlu.user_acts = []
	nlu._match_general_act(user_utterance="hello")
	assert nlu.user_acts == [UserAct("hello", UserActionType.Hello, None, None, 1.0)]



def test_match_general_act_bye(nlu):
	"""

	Tests if general act bye are matched properly

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	nlu.user_acts = []
	nlu._match_general_act(user_utterance="bye")
	expected_user_act = [UserAct("hello", UserActionType.Bye, None, None, 1.0)]
	assert nlu.user_acts == expected_user_act




def test_dontcare_in_general_regexes(nlu):
	"""

	Tests whether 'dont care' is recognized

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	assert "dontcare" in nlu.general_regex



def test_match_general_act_dontcare_empty(nlu):
	"""

	Tests if general act dontcare is empty without further specification

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	nlu.user_acts = []
	nlu._match_general_act(user_utterance="dontcare")
	assert nlu.user_acts == []



def test_match_general_act_dontcare(nlu, domain):
    """

    Tests if general act dontcare is matched properly

    Args:
        nlu: NlU Object (given in conftest.py)

    """
    nlu.__init__(domain)
    nlu.user_acts = []
    nlu.sys_act_info['last_act'] = SysAct(act_type=SysActionType.Request, slot_values={"primary_uniform_color": []})

    nlu.lastRequestSlot = "primary_uniform_color"

    nlu._match_general_act(user_utterance="I dont care")
    expected_user_act = UserAct("", act_type=UserActionType.Inform, slot='primary_uniform_color', value="dontcare", score=1.0)
    assert nlu.user_acts[0] == expected_user_act



def test_match_domain_specific_act_inform(nlu):
	"""

	Tests whether a user's inform statement is recognized and stored properly

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	nlu.user_acts = []
	nlu.slots_informed = set()
	nlu._match_domain_specific_act(user_utterance='the uniform should be Purple')
	assert 'primary_uniform_color' in nlu.slots_informed
	assert nlu.slots_informed == {'primary_uniform_color'}
	assert type(nlu.slots_informed) == set



def test_match_domain_specific_act_reqquest(nlu):
	"""

	Tests whether a user request is recognized and stored properly

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	nlu.user_acts = []
	nlu.slots_requested = set()
	nlu._match_domain_specific_act(user_utterance='what is their location')
	assert 'last_known_location' in nlu.slots_requested
	assert nlu.slots_requested == {'last_known_location'}
	assert type(nlu.slots_requested) == set



def test_check(nlu):
	"""

	Tests if matching of user utterance and and user expression is recognized as matching and non matching

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	user_utterance = "Hi"
	act = "hello"
	assert nlu._check(None) == False
	assert nlu._check(re.search(nlu.general_regex[act], user_utterance, re.I)) == True



def test_assigned_score(nlu):
	"""

	Tests if assigned score is 1.0 

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	act = UserAct()
	assert act.score == 1.0



def test_start_dialog(nlu):
	"""

	Tests whether all previous system acts where set to None

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	nlu.sys_act_info = {'foo':'ba'}
	nlu.start_dialog()
	assert nlu.sys_act_info == {'last_act': None, 'lastInformedPrimKeyVal': None, 'lastRequestSlot': None}



def test_disambiguate_co_occurrence():
	"""
	
	# TODO UNSURE WHAT AND HOW TO TEST SINCE FUNCTION IS USELESS, I.E. NEVER CALLED, AND CAN BE DELETED


	"""
	pass



def test_solve_informable_values(nlu):
	"""

	# TODO UNSURE WHAT AND HOW TO TEST SINCE FUNCTION IS USELESS, I.E. NEVER CALLED, AND CAN BE DELETED

	"""
	pass



def test_initializing_language_is_english_by_default(nlu):
	"""

	Test if language is set to English by default

	Args:
		nlu: NlU Object (given in conftest.py)

	"""
	nlu._initialize()
	assert nlu.language == Language.ENGLISH
	assert nlu.language != Language.GERMAN