import os
import sys
import pytest

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(get_root_dir())
from services.service import Service, PublishSubscribe
from utils.topics import Topic


@pytest.fixture
def serviceA(domain_name):
    class ServiceA(Service):

        def __init__(self, domain):
            Service.__init__(self, domain=domain)


        @PublishSubscribe(pub_topics=['topic1'])
        def send_message(self):
            return {'topic1': 'Hello'}
    return ServiceA(domain=domain_name)


@pytest.fixture
def serviceB(domain_name):
    class ServiceB(Service):

        def __init__(self, domain):
            Service.__init__(self, domain=domain)

        @PublishSubscribe(sub_topics=['topic1'], pub_topics=['topic2'])
        def subpub(self, topic1):
            return {'topic2': 'World!'}
    return ServiceB(domain=domain_name)


@pytest.fixture
def serviceC(domain_name):
    class ServiceC(Service):

        def __init__(self, domain):
            Service.__init__(self, domain=domain)

        @PublishSubscribe(sub_topics=['topic1', 'topic2'], pub_topics=[Topic.DIALOG_END])
        def concatenate(self, topic1, topic2):
            print(f"{topic1} {topic2}")
            return {Topic.DIALOG_END: True}
    return ServiceC(domain=domain_name)


@pytest.fixture
def serviceC_with_inconsistent_subscriber(domain_name):
    class ServiceC(Service):

        def __init__(self, domain):
            Service.__init__(self, domain=domain)

        @PublishSubscribe(sub_topics=['topic1', 'topic2', 'topic3'])
        def concatenate(self, topic1, topic2, topic3):
            print(f"{topic1} {topic2} {topic3}")
            return {Topic.DIALOG_END: True}
    return ServiceC(domain=domain_name)



@pytest.fixture
def serviceC_with_inconsistent_publisher(domain_name):
    class ServiceC(Service):

        def __init__(self, domain):
            Service.__init__(self, domain=domain)

        @PublishSubscribe(sub_topics=['topic1', 'topic2'], pub_topics=[Topic.DIALOG_END, 'topic3'])
        def concatenate(self, topic1, topic2):
            print(f"{topic1} {topic2}")
            return {Topic.DIALOG_END: True, 'topic3': None}
    return ServiceC(domain=domain_name)

