import os
import sys
import argparse

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.domain_tracker import DomainTracker

# Very simple test cases

def test_reset_turn_count_on_start():
    tracker = DomainTracker([])
    tracker.turn = 27
    tracker.dialog_start()
    assert tracker.turn == 0

def test_reset_domain_on_start():
    tracker = DomainTracker([])
    tracker.current_domain = "Balhdka"
    tracker.dialog_start()
    assert tracker.current_domain is None
