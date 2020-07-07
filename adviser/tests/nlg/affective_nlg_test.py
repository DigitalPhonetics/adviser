import os
import sys
import argparse
import pytest


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
from services.nlg import HandcraftedNLG
from services.nlg.affective_nlg import HandcraftedEmotionNLG
from utils.common import Language
from services.service import Service
from utils.sysact import SysAct, SysActionType
from utils.domain.jsonlookupdomain import JSONLookupDomain




def test_init(aff_nlg, domain):
	aff_nlg.__init__(domain)
	Service.__init__(aff_nlg, domain='superhero')
	
	assert aff_nlg.domain == 'superhero'