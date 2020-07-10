import os
import sys
import argparse
import pytest


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
from services.nlg import HandcraftedNLG
from utils.common import Language
from services.service import Service
from utils.sysact import SysAct, SysActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain



def test_init(nlg, domain):
	"""

	Tests whether given class is initialized properly

	Args:
		nlg: NLG Object (given in conftest.py)
		domain: Domain Object (given in conftest.py)

	"""
	nlg.__init__(domain)
	Service.__init__(nlg, domain='superhero')

	assert nlg.language == Language.ENGLISH
	assert nlg.template_english == None
	assert nlg.template_german == None
	assert nlg.domain == 'superhero'
	assert 'superheroMessages.nlg' in nlg.template_filename



def test_publish_system_utterance(nlg):
	"""

	Tests whether the format is returned as exprected

	Args:
		nlg: NLG Object (given in conftest.py)

	"""	
	sys_act = SysAct(act_type=SysActionType.Welcome)
	msg_out = nlg.generate_system_utterance(sys_act)
	expected_return = {'sys_utterance': 'Welcome to the SuperherO Support (SOS)  chat bot. How may I help you?'}
	assert 'sys_utterance' in expected_return.keys()
	assert msg_out in expected_return.values()



def test_generate_system_utterance_welcomemsg(nlg):
	"""
	
	Tests whether the correct welcome message is chosen from the given domain
	
	Args:
		nlg: NLG Object (given in conftest.py)

	"""
	sys_act = SysAct(act_type=SysActionType.Welcome)
	msg_out = nlg.generate_system_utterance(sys_act, sys_emotion='Happy', sys_engagement='low')
	expected_message = 'Welcome to the SuperherO Support (SOS)  chat bot. How may I help you?'
	assert type(msg_out) == str
	assert msg_out == expected_message



def test_generate_system_utterance_bad(nlg):
	"""

	Tests whether the correct template is seletected for a bad system act
	
	Args:
		nlg: NLG Object (given in conftest.py)

	"""
	sys_act = SysAct(act_type=SysActionType.Bad)
	msg_out = nlg.generate_system_utterance(sys_act)
	expected_message = 'Sorry I am a bit confused; please tell me again what you are looking for.'
	assert type(msg_out) == str
	assert msg_out == expected_message



def test_generate_system_utterance_request_primary_uniform_colour(nlg):
	"""

	Tests exemplary whether the right template is chosen for a given request setup

	Args:
		nlg: NLG Object (given in conftest.py)

	"""
	sys_act = SysAct(act_type=SysActionType.Request, slot_values={"primary_uniform_color": []})
	msg_out = nlg.generate_system_utterance(sys_act)
	expected_message = 'What should be the primary color of the superhero\'s uniform?'
	assert type(msg_out) == str
	assert msg_out == expected_message



def test_initialize_language_english(nlg):
	"""

	Tests whether given class is initialized properly

	Args:
		nlg: NLG Object (given in conftest.py)

	"""
	nlg._initialise_language(language=Language.ENGLISH)
	assert nlg.template_filename != None
	assert nlg.template_english == None
	assert 'superheroMessages' in nlg.template_filename



def test_initialize_language_german_courses(nlg):
	"""

	Tests whether given class is initialized properly

	Args:
		nlg: NLG Object (given in conftest.py)

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlg = HandcraftedNLG(domain)
	nlg._initialise_language(language=Language.GERMAN)
	assert nlg.template_filename != None
	assert nlg.template_english == None
	assert nlg.template_german == None
	assert 'ImsCoursesMessagesGerman' in nlg.template_filename