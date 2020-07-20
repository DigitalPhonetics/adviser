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



def test_setup_of_user_informables_courses():
	"""
    
    Tests if user informable slots are recognized as such

    """
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)

	for user_informable_slot in ['applied_nlp', 'bachelor', 'cognitive_science', 'course_type', 'deep_learning', 'ects',
								'elective', 'extendable', 'lecturer', 'lang', 'linguistics', 'machine_learning', 'master', 
								'module_id', 'module_name','name', 'obligatory_attendance', 'oral_exam', 'participation_limit', 
								'presentation', 'programming', 'project', 'report', 'semantics', 'speech', 'statistics', 'sws',
								'syntax', 'turn', 'written_exam']:
		assert user_informable_slot in nlu.USER_INFORMABLE



def test_setup_of_user_requestables_courses():
	"""

    Tests if user requestable slots are recognized as such

    """
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)

	for user_requestable_slot in ['applied_nlp', 'bachelor', 'cognitive_science', 'course_type', 'deep_learning', 'ects',
								'elective', 'extendable', 'lecturer', 'lang', 'linguistics', 'machine_learning', 'master', 
								'module_id', 'module_name','name', 'obligatory_attendance', 'oral_exam', 'participation_limit', 
								'presentation', 'programming', 'project', 'report', 'semantics', 'speech', 'statistics', 'sws',
								'syntax', 'turn', 'written_exam']:
		assert user_requestable_slot in nlu.USER_REQUESTABLE



def test_inform_courses_course_type():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot-value pair

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Inform
	act_out.slot = "course_type"
	act_out.value = "se"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='seminar')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



def test_inform_courses_language():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot-value pair

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
	   
	act_out = UserAct()
	act_out.type = UserActionType.Inform
	act_out.slot = "lang"
	act_out.value = "de"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='german')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



def test_inform_courses_course_type_sentence():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot-value pair

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Inform
	act_out.slot = "course_type"
	act_out.value = "ue"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='I am looking for a practical course')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



# test binary inform

def test_inform_courses_binary_not_linguistics():
	"""
	
  	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Inform
	act_out.slot = "linguistics"
	act_out.value = "false"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='not linguistics')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



def test_inform_courses_binary_written_exam():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Inform
	act_out.slot = "written_exam"
	act_out.value = "true"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='written exam')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



def test_inform_courses_binary_bachelor():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Inform
	act_out.slot = "bachelor"
	act_out.value = "true"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='i want a bachelor course')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



def test_inform_courses_binary_deep_learning():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Inform
	act_out.slot = "deep_learning"
	act_out.value = "false"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='not neural networks')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



def test_inform_courses_binary_extendable():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain binary slot-value pair

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Inform
	act_out.slot = "extendable"
	act_out.value = "true"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='i want an extendable course')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



# test request

def test_request_name():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Request
	act_out.slot = "name"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='what is it called')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



def test_request_ects():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Request
	act_out.slot = "ects"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='how many credits')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



def test_request_time_slot():
	"""
	
	Tests exemplary whether a given synonym, i.e. a user utterance, is recognized as belonging to a certain slot

	"""
	domain = JSONLookupDomain('ImsCourses')
	nlu = HandcraftedNLU(domain)
   
	act_out = UserAct()
	act_out.type = UserActionType.Request
	act_out.slot = "time_slot"

	usr_utt = nlu.extract_user_acts(nlu, user_utterance='at what time')
	assert 'user_acts' in usr_utt
	assert usr_utt['user_acts'][0] == act_out



# test for multiple user acts

def test_multiple_user_acts_courses():
    """
    
    Tests exemplary whether a given sentence with multiple user acts is understood properly
    
    """
    domain = JSONLookupDomain('ImsCourses')
    nlu = HandcraftedNLU(domain)
    
    usr_utt = nlu.extract_user_acts(nlu, user_utterance='Hi, I want a course that is related to linguistics')
    assert 'user_acts' in usr_utt

    act_out = UserAct()
    act_out.type = UserActionType.Hello
    assert usr_utt['user_acts'][0] == act_out

    act_out = UserAct()
    act_out.type = UserActionType.Inform
    act_out.slot = "linguistics"
    act_out.value = "true"
    assert usr_utt['user_acts'][1] == act_out