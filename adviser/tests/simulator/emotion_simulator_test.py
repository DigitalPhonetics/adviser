import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.simulator import EmotionSimulator
from utils.userstate import EmotionType, EngagementType

def execute_send_emotion(emotion_simulator):
    """
    Tests the return value of the send_emotion method. Should be called in each test handling
    send_emotion to make sure that it returns in every case the same structure.

    Args:
        emotion_simulator: EmotionSimulator Object

    Returns:
        (dict): a dictionary containing emotion and engagement
    """
    emotion_dict = emotion_simulator.send_emotion()
    assert type(emotion_dict) == dict
    assert len(emotion_dict) == 2
    assert 'emotion' in emotion_dict
    assert 'engagement' in emotion_dict
    assert type(emotion_dict['emotion']) == dict
    assert 'category' in emotion_dict['emotion']
    return emotion_dict


def test_send_random_emotion():
    """
    Tests sending a random emotion.
    """
    emotion_simulator = EmotionSimulator(random=True)
    emotion_dict = execute_send_emotion(emotion_simulator)
    assert isinstance(emotion_dict['emotion']['category'], EmotionType)
    assert isinstance(emotion_dict['engagement'], EngagementType)


@pytest.mark.parametrize('static_emotion', [emotion for emotion in EmotionType])
@pytest.mark.parametrize('static_engagement', [engagement for engagement in EngagementType])
def test_send_static_emotion(static_emotion, static_engagement):
    """
    Tests sending a static emotion and engagement. Tests all possible combinations of EmotionType and EngagementType.

    Args:
        static_emotion: a possible EmotionType
        static_engagement: a possible EngagementType
    """
    emotion_simulator = EmotionSimulator(random=False, static_emotion=static_emotion,
                                         static_engagement=static_engagement)
    emotion_dict = execute_send_emotion(emotion_simulator)
    assert emotion_dict['emotion']['category'] == static_emotion
    assert emotion_dict['engagement'] == static_engagement
