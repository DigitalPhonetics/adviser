import os
import sys
import pickle
from time import sleep
import zmq
from zmq.devices import ProcessProxy
from threading import Thread
from queue import Queue
import pytest


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.service import _send_msg, _recv_ack, PublishSubscribe, Service

# auxiliary functions

def setup_service(service):
    """
    Sets up a service by initializing its publisher and subscriber network, including a proxy to
    allow the communication between them.

    Args:
        service (Service): Instance of a class that inherits of the Service class.

    Return:
        service (Service): The initialized service.
    """
    proxy = ProcessProxy(in_type=zmq.XSUB, out_type=zmq.XPUB)
    proxy.bind_in(f"{service._protocol}://{service._host_addr}:{service._pub_port}")
    proxy.bind_out(f"{service._protocol}://{service._host_addr}:{service._sub_port}")
    proxy.start()

    service._init_pubsub()
    service._register_with_dialogsystem()

    return service


def teardown_service(service):
    """
    Tears down a service by sending terminate messages to all active subscribers and closing the
    sockets.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    ctx = zmq.Context.instance()
    ctrl_publisher = ctx.socket(zmq.PUB)
    ctrl_publisher.sndhwm = 1100000
    ctrl_publisher.connect(f"{service._protocol}://{service._host_addr}:{service._pub_port}")

    ctrl_subscriber = ctx.socket(zmq.SUB)
    ctrl_subscriber.setsockopt(zmq.SUBSCRIBE, bytes(f"ACK/{service._terminate_topic}", encoding="ascii"))
    ctrl_subscriber.connect(f"{service._protocol}://{service._host_addr}:{service._sub_port}")
    sleep(.25)

    # close all subscriber sockets and terminate their threads
    _send_msg(ctrl_publisher, service._terminate_topic, True)
    _recv_ack(ctrl_subscriber, service._terminate_topic, True)


    # close all publisher sockets
    ctrl_publisher.close()
    ctrl_subscriber.close()
    for publisher in service._publish_sockets.values():
        publisher.close()


def recv_all_messages(q, protocol, host, sub_port):
    """
    Receives all messages that are sent to any topics.

    Args:
        q (Queue): Queue to store the received messages in.
        protocol (str): Communication protocol.
        host (str): IP-address of the host (e.g. localhost).
        sub_port (int): Subscriber port.
    """
    ctx = zmq.Context.instance()
    subscriber = ctx.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')  # subscribe to all topics
    subscriber.connect(f"{protocol}://{host}:{sub_port}")
    sleep(.1)
    while True:
        msg = subscriber.recv_multipart(copy=True)
        if msg[0].decode('ascii') == 'stop_test':  # condition to stop the thread
            break
        q.put(msg)

    subscriber.close()

# test functions

def test_PublishSubscribe_without_parameters():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator without
    given topics does not fail on execution.
    """
    class AService(Service):
        def __init__(self):
            Service.__init__(self)

        @PublishSubscribe()
        def a_function(self):
            return {}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)

    # execute function on explicit call
    res = service.a_function()

    sleep(.25)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == list()
    assert getattr(service.a_function, 'queued_sub_topics') == list()
    assert getattr(service.a_function, 'pub_topics') == list()
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert res == dict()
    assert recv_messages == dict()


def test_PublishSubscribe_with_subscribed_topics_on_explicit_call():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator with given
    subscribed topics works correctly when the function is called explicitly.
    """
    sub_topic1 = 'topic1'
    sub_content1 = 'foo'
    sub_topic2 = 'topic2'
    sub_content2 = 'bar'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)
            self.content1 = None
            self.content2 = None

        @PublishSubscribe(sub_topics=[sub_topic1, sub_topic2])
        def a_function(self, topic1, topic2):
            self.content1 = topic1
            self.content2 = topic2
            return {}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)

    # execute function on explicit call
    res = service.a_function(topic1=sub_content1, topic2=sub_content2)

    sleep(.25)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == [sub_topic1, sub_topic2]
    assert getattr(service.a_function, 'queued_sub_topics') == list()
    assert getattr(service.a_function, 'pub_topics') == list()
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert res == dict()
    assert recv_messages == dict()
    assert service.content1 == sub_content1
    assert service.content2 == sub_content2


def test_PublishSubscribe_with_subscribed_topics_on_implicit_call():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator with given
    subscribed topics works correctly when the function is called automatically. The automatic
    function call happens when messages for all subscribed topics have been sent by a service.
    """
    sub_topic1 = 'topic1'
    sub_content1 = 'foo'
    sub_topic2 = 'topic2'
    sub_content2 = 'bar'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)
            self.content1 = None
            self.content2 = None

        @PublishSubscribe(sub_topics=[sub_topic1, sub_topic2])
        def a_function(self, topic1, topic2):
            self.content1 = topic1
            self.content2 = topic2
            return {}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)
    _send_msg(service._control_channel_pub, service._start_topic, True)
    sleep(.1)

    # trigger execution of a_function by sending messages to its subscribed topics
    _send_msg(service._control_channel_pub, sub_topic1, sub_content1)
    _send_msg(service._control_channel_pub, sub_topic2, sub_content2)

    sleep(.5)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == [sub_topic1, sub_topic2]
    assert getattr(service.a_function, 'queued_sub_topics') == list()
    assert getattr(service.a_function, 'pub_topics') == list()
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert recv_messages != dict()
    assert service.content1 == sub_content1
    assert service.content2 == sub_content2


def test_PublishSubscribe_with_subscribed_topics_receives_only_most_recent_messages():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator with given
    subscribed topics receives for each topic only the most recent message.
    """
    sub_topic1 = 'topic1'
    sub_content1 = 'foo'
    sub_topic2 = 'topic2'
    sub_content2 = 'bar'
    sub_content3 = 'baz'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)
            self.content1 = None
            self.content2 = None

        @PublishSubscribe(sub_topics=[sub_topic1, sub_topic2])
        def a_function(self, topic1, topic2):
            self.content1 = topic1
            self.content2 = topic2
            return {}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)
    _send_msg(service._control_channel_pub, service._start_topic, True)
    sleep(.1)

    # trigger execution of a_function by sending messages to its subscribed topics
    _send_msg(service._control_channel_pub, sub_topic1, sub_content1)
    _send_msg(service._control_channel_pub, sub_topic1, sub_content3)
    _send_msg(service._control_channel_pub, sub_topic2, sub_content2)

    sleep(.5)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == [sub_topic1, sub_topic2]
    assert getattr(service.a_function, 'queued_sub_topics') == list()
    assert getattr(service.a_function, 'pub_topics') == list()
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert recv_messages != dict()
    assert service.content1 != sub_content1
    assert service.content1 == sub_content3
    assert service.content2 == sub_content2


def test_PublishSubscribe_with_published_topics():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator with given
    published topics works correctly. The function will both return its return value (a
    dictionary of published topics and respective content) and send it as messages.
    """
    pub_topic1 = 'topic1'
    pub_content1 = 'foo'
    pub_topic2 = 'topic2'
    pub_content2 = 'bar'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)

        @PublishSubscribe(pub_topics=[pub_topic1, pub_topic2])
        def a_function(self):
            return {pub_topic1: pub_content1, pub_topic2: pub_content2}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)

    # execute function on explicit call
    res = service.a_function()

    sleep(.25)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == list()
    assert getattr(service.a_function, 'queued_sub_topics') == list()
    assert getattr(service.a_function, 'pub_topics') == [pub_topic1, pub_topic2]
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert isinstance(res, dict)
    assert len(res) == 2
    assert res == {pub_topic1: pub_content1, pub_topic2: pub_content2}
    assert res.items() <= recv_messages.items()


def test_PublishSubscribe_with_published_topics_selection():
    """
     Tests whether a function that is decorated with the PublishSubscribe decorator with given
     published topics works correctly even if it does not return a content for each of its
     published topics.
     """
    pub_topic1 = 'topic1'
    pub_content1 = 'foo'
    pub_topic2 = 'topic2'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)

        @PublishSubscribe(pub_topics=[pub_topic1, pub_topic2])
        def a_function(self):
            return {pub_topic1: pub_content1}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)

    # execute function on explicit call
    res = service.a_function()

    sleep(.25)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == list()
    assert getattr(service.a_function, 'queued_sub_topics') == list()
    assert getattr(service.a_function, 'pub_topics') == [pub_topic1, pub_topic2]
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert isinstance(res, dict)
    assert len(res) == 1
    assert res == {pub_topic1: pub_content1}
    assert res.items() <= recv_messages.items()

def test_PublishSubscribe_with_subscribed_and_published_topics_on_explicit_call():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator with given
    subscribed topics and published topics works correctly when the function is called explicitly.
    """
    sub_topic = 'topic1'
    sub_content = 'foo'
    pub_topic = 'topic2'
    pub_content = 'bar'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)
            self.content = None

        @PublishSubscribe(sub_topics=[sub_topic], pub_topics=[pub_topic])
        def a_function(self, topic1):
            self.content = topic1
            return {pub_topic: pub_content}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)

    # execute function on explicit call
    res = service.a_function(sub_content)

    sleep(.25)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == [sub_topic]
    assert getattr(service.a_function, 'queued_sub_topics') == list()
    assert getattr(service.a_function, 'pub_topics') == [pub_topic]
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert isinstance(res, dict)
    assert len(res) == 1
    assert pub_topic in res
    assert res[pub_topic] == pub_content
    assert res.items() <= recv_messages.items()
    assert service.content == sub_content


def test_PublishSubscribe_with_subscribed_and_published_topics_on_implicit_call():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator with given
    subscribed topics and published topics works correctly when the function is called
    automatically. The automatic function call happens when messages for all subscribed topics
    have been sent by a service.
    """
    sub_topic = 'topic1'
    sub_content = 'foo'
    pub_topic = 'topic2'
    pub_content = 'bar'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)
            self.content = None

        @PublishSubscribe(sub_topics=['topic1'], pub_topics=[pub_topic])
        def a_function(self, topic1):
            self.content = topic1
            return {pub_topic: pub_content}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)
    _send_msg(service._control_channel_pub, service._start_topic, True)
    sleep(.1)

    # trigger execution of a_function by sending a message to its subscribed topic
    _send_msg(service._control_channel_pub, sub_topic, sub_content)

    sleep(.5)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == [sub_topic]
    assert getattr(service.a_function, 'queued_sub_topics') == list()
    assert getattr(service.a_function, 'pub_topics') == [pub_topic]
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert pub_topic in recv_messages
    assert recv_messages[pub_topic] == pub_content
    assert service.content == sub_content


def test_PublishSubscribe_with_queued_topics_on_explicit_call():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator with given
    queued subscribed topics works correctly when the function is called
    explicitly.
    """
    queued_topic1 = 'topic1'
    queued_content1 = 'foo'
    queued_topic2 = 'topic2'
    queued_content2 = 'bar'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)
            self.content1 = None
            self.content2 = None

        @PublishSubscribe(queued_sub_topics=[queued_topic1, queued_topic2])
        def a_function(self, topic1, topic2):
            self.content1 = topic1
            self.content2 = topic2
            return {}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)

    # execute function on explicit call
    res = service.a_function(topic1=queued_content1, topic2=queued_content2)

    sleep(.25)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == list()
    assert getattr(service.a_function, 'queued_sub_topics') == [queued_topic1, queued_topic2]
    assert getattr(service.a_function, 'pub_topics') == list()
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert res == dict()
    assert recv_messages == dict()
    assert service.content1 == queued_content1
    assert service.content2 == queued_content2


def test_PublishSubscribe_with_queued_topics_on_implicit_call():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator with given
    queued subscribed topics works correctly when the function is called
    automatically. The automatic function call happens when messages for all subscribed topics
    have been sent by a service.
    """
    queued_topic1 = 'topic1'
    queued_content1 = 'foo'
    queued_topic2 = 'topic2'
    queued_content2 = 'bar'

    class AService(Service):
        def __init__(self):
            Service.__init__(self)
            self.content1 = None
            self.content2 = None

        @PublishSubscribe(queued_sub_topics=[queued_topic1, queued_topic2])
        def a_function(self, topic1, topic2):
            self.content1 = topic1
            self.content2 = topic2
            return {}

    q = Queue()
    service = setup_service(AService())
    thread = Thread(target=recv_all_messages, args=(q, service._protocol, service._host_addr,
                                                    service._sub_port))
    thread.start()
    sleep(.25)
    _send_msg(service._control_channel_pub, service._start_topic, True)
    sleep(.1)

    # trigger execution of a_function by sending messages to its subscribed topics
    _send_msg(service._control_channel_pub, queued_topic1, queued_content1)
    _send_msg(service._control_channel_pub, queued_topic2, queued_content2)

    sleep(.5)
    _send_msg(service._control_channel_pub, 'stop_test', True)
    thread.join(5)
    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    teardown_service(service)

    assert getattr(service.a_function, 'pubsub') is True
    assert getattr(service.a_function, 'sub_topics') == list()
    assert getattr(service.a_function, 'queued_sub_topics') == [queued_topic1, queued_topic2]
    assert getattr(service.a_function, 'pub_topics') == list()
    assert getattr(service.a_function, 'timestamp_enabled') is False
    assert recv_messages != dict()
    assert service.content1 == [queued_content1]
    assert service.content2 == [queued_content2]


def test_PublishSubscribe_fails_if_returns_no_dict():
    """
    Tests whether a function that is decorated with the PublishSubscribe decorator fails when it
    does not return a dictionary.
    """
    class AService(Service):
        def __init__(self):
            Service.__init__(self)

        @PublishSubscribe(pub_topics=['topic'])
        def a_function(self):
            return 'foo'

    service = setup_service(AService())

    with pytest.raises(TypeError):
        service.a_function()      # execute function on explicit call
    teardown_service(service)
