import os
import sys
import threading
import zmq
from zmq.devices import ProcessProxy
from time import sleep
import pytest
from queue import Queue
import pickle


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.service import Service, _send_msg, _recv_ack, PublishSubscribe


# auxiliary service class
class ServiceA(Service):

    def __init__(self, *args, **kwargs):
        """
        Auxiliary service with a dummy function.
        """
        Service.__init__(self, *args, **kwargs)
        self.dummy_function_result = None

    @PublishSubscribe(sub_topics=['foo'], queued_sub_topics=['bar'], pub_topics=['baz'])
    def dummy_function(self, *args, **kwargs):
        """
        Dummy function, decorated with PublishSubscribe, that just stores the input in a variable.
        """
        self.dummy_function_result = kwargs
        return {'baz': True}


# auxiliary functions

@pytest.fixture
def service(request):
    """
    Creates a service instance and initializes a proxy. Once a test using this resource has
    finished, the service will be torn down again by terminating all of its threads and closing
    its sockets.

    Args:
        request: A request object containing the context of a test.
    """
    # setup service
    service = ServiceA()

    _proxy_dev = ProcessProxy(in_type=zmq.XSUB, out_type=zmq.XPUB)
    _proxy_dev.bind_in(f"{service._protocol}://{service._host_addr}:{service._pub_port}")
    _proxy_dev.bind_out(f"{service._protocol}://{service._host_addr}:{service._sub_port}")
    _proxy_dev.start()

    yield service

    # teardown service
    # get list of terminate topics that should be sent to terminate the respective threads
    # each test function is marked with their list of terminate topics
    marker = request.node.get_closest_marker("terminate_topics")
    if marker is None:
        terminate_topics = []
    else:
        variable = marker.args[0]
        terminate_topics = getattr(service, variable, [])
        if isinstance(terminate_topics, dict):
            terminate_topics = list(terminate_topics.keys())
        elif not isinstance(terminate_topics, list):
            terminate_topics = [terminate_topics]

    ctx = zmq.Context.instance()
    ctrl_publisher = ctx.socket(zmq.PUB)
    ctrl_publisher.sndhwm = 1100000
    ctrl_publisher.connect(f"{service._protocol}://{service._host_addr}:{service._pub_port}")

    ctrl_subscriber = ctx.socket(zmq.SUB)
    for topic in terminate_topics:
        ctrl_subscriber.setsockopt(zmq.SUBSCRIBE, bytes(f"ACK/{topic}", encoding="ascii"))

    ctrl_subscriber.connect(f"{service._protocol}://{service._host_addr}:{service._sub_port}")
    sleep(.25)

    # close all subscriber sockets and terminate their threads
    for topic in terminate_topics:
        print(topic)
        _send_msg(ctrl_publisher, topic, True)
        _recv_ack(ctrl_subscriber, topic, True)

    # close all publisher sockets
    ctrl_publisher.close()
    ctrl_subscriber.close()
    for publisher in service._publish_sockets.values():
        publisher.close()


def get_all_messages_to_published_topic(topics, protocol, host, pub_port, sub_port):
    """
    Receives all messages that are published when publishing to a given set of topics. For
    instance, when publishing to the terminate topic of a service, an acknowledgement of this
    termination should be published after that.

    Args:
        topics (list): either a list of single topics or of (topic, content) pairs.
        protocol (str): Communication protocol.
        host (str): IP-address of the host (e.g. localhost).
        pub_port (int): Publisher port.
        sub_port (int): Subscriber port.

    Returns:
         Queue containing all received messages.
    """
    ctx = zmq.Context.instance()
    publisher = ctx.socket(zmq.PUB)
    publisher.sndhwm = 1100000
    publisher.connect(f"{protocol}://{host}:{pub_port}")

    q = Queue()

    def recv_all_messages(q):
        ctx = zmq.Context.instance()
        subscriber = ctx.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b'')  # subscribe to all topics
        subscriber.connect(f"{protocol}://{host}:{sub_port}")
        sleep(.1)
        while True:
            msg = subscriber.recv_multipart(copy=True)
            if msg[0].decode('ascii') == 'stop_test':
                break
            q.put(msg)

        subscriber.close()

    thread = threading.Thread(target=recv_all_messages, args=(q,))
    thread.start()
    sleep(.25)
    for topic in topics:
        if isinstance(topic, tuple):
            _send_msg(publisher, topic[0], topic[1])
        else:
            _send_msg(publisher, topic, True)
    sleep(.5)
    _send_msg(publisher, 'stop_test', True)
    thread.join(timeout=5)
    return q


# test functions

@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_init_pubsub_creates_listener(service):
    """
    Tests whether the initialization of publishers and subscribers creates listener. In this
    case, new threads (one for each listener) should have been spawned.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._sub_topics = set()
    threads = threading.enumerate()
    service._init_pubsub()
    assert service._sub_topics != set()
    assert threading.active_count() > len(threads)


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_init_pubsub_creates_publishers(service):
    """
    Tests whether the initialization of publishers and subscribers creates publisher.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._pub_topics = set()
    service._publish_sockets = dict()
    service._init_pubsub()
    assert service._pub_topics != set()
    assert service._publish_sockets != dict()


@pytest.mark.terminate_topics('_terminate_topic')
def test_register_with_dialogsystem(service):
    """
    Tests whether registering the service with a dialog system creates control sockets.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    threads = threading.enumerate()
    service._register_with_dialogsystem()
    assert threading.active_count() > len(threads)
    assert hasattr(service, '_control_channel_sub')
    assert hasattr(service, '_control_channel_pub')
    assert hasattr(service, '_internal_control_channel_sub')


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_setup_listener_without_subscriptions(service):
    """
    Tests whether setting up listeners without providing a list of subscriptions does not create
    any listeners (i.e. no new threads are spawned).

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    func_instance = service.dummy_function
    topics, queued_topics = [], []
    service._sub_topics = set()
    threads = threading.enumerate()

    service._setup_listener(func_instance, topics, queued_topics)
    assert service._sub_topics == set()
    assert threading.active_count() == len(threads)


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_setup_listener_with_subscriptions(service):
    """
    Tests whether setting up listeners with providing a list of subscriptions will create new
    threads and register start, end and terminate topics for the listeners.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    func_instance = service.dummy_function
    topics, queued_topics = ['foo'], ['bar']
    service._sub_topics = set()
    threads = threading.enumerate()

    service._setup_listener(func_instance, topics, queued_topics)
    assert service._sub_topics == set(topics) | set(queued_topics)
    assert threading.active_count() > len(threads)
    assert f"{str(func_instance)}/START" in service._internal_start_topics
    assert service._internal_start_topics[f"{str(func_instance)}/START"] == str(func_instance)
    assert f"{str(func_instance)}/END" in service._internal_end_topics
    assert service._internal_end_topics[f"{str(func_instance)}/END"] == str(func_instance)
    assert f"{str(func_instance)}/TERMINATE" in service._internal_terminate_topics
    assert service._internal_terminate_topics[f"{str(func_instance)}/TERMINATE"] == str(
        func_instance)


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_setup_publishers_without_topics(service):
    """
    Tests whether setting up publishers without providing any topics will not create any
    publishers.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    func_instance = service.dummy_function
    topics = []

    service._pub_topics = set()
    service._publish_sockets = dict()
    service._setup_publishers(func_instance, topics)
    assert service._pub_topics == set()
    assert service._publish_sockets == dict()


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_setup_publishers_with_topics(service):
    """
    Tests whether setting up publishers with providing topics will create publisher sockets.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    func_instance = service.dummy_function
    topics = ['baz']

    service._pub_topics = set()
    service._publish_sockets = dict()
    service._setup_publishers(func_instance, topics)
    assert service._pub_topics == set(topics)
    assert service._publish_sockets != dict()
    assert all(isinstance(socket, zmq.Socket) for socket in service._publish_sockets.values())
    assert all(socket.getsockopt(zmq.TYPE) == 1 for socket in service._publish_sockets.values())
    # Type 1 = PUB


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_setup_dialog_ctrl_msg_listener(service):
    """
    Tests whether setting up a dialog control message listener will create subscribers and
    publishers for controlling the messages coming from and to the dialog system.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._setup_dialog_ctrl_msg_listener()
    assert hasattr(service, '_control_channel_sub')
    assert isinstance(service._control_channel_sub, zmq.Socket)
    assert service._control_channel_sub.getsockopt(zmq.TYPE) == 2 # type 2 = SUB
    assert hasattr(service, '_control_channel_pub')
    assert isinstance(service._control_channel_pub, zmq.Socket)
    assert service._control_channel_pub.getsockopt(zmq.TYPE) == 1  # type 1 = PUB
    assert hasattr(service, '_internal_control_channel_sub')
    assert isinstance(service._internal_control_channel_sub, zmq.Socket)
    assert service._internal_control_channel_sub.getsockopt(zmq.TYPE) == 2  # type 2 = SUB


@pytest.mark.terminate_topics('_terminate_topic')
def test_control_channel_listener_for_start_topic(service):
    """
    Tests whether the control channel listener sends a start topic to all internal subscribers
    when it receives the service start topic.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    service._register_with_dialogsystem()
    expected_topics = [service._start_topic] + list(service._internal_start_topics.keys())
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]

    q = get_all_messages_to_published_topic(topics=[service._start_topic],
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)


@pytest.mark.terminate_topics('_terminate_topic')
def test_control_channel_listener_for_end_topic(service):
    """
    Tests whether the control channel listener sends an end topic to all internal subscribers
    when it receives the service end topic.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    service._register_with_dialogsystem()
    expected_topics = [service._end_topic] + list(service._internal_end_topics.keys())
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]

    q = get_all_messages_to_published_topic(topics=[service._end_topic],
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)


def test_control_channel_listener_for_terminate_topic(service):
    """
    Tests whether the control channel listener sends a terminate topic to all internal subscribers
    when it receives the service terminate topic.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    service._register_with_dialogsystem()
    expected_topics = [service._terminate_topic] + list(service._internal_terminate_topics.keys())
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]

    q = get_all_messages_to_published_topic(topics=[service._terminate_topic],
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)


@pytest.mark.terminate_topics('_terminate_topic')
def test_control_channel_listener_for_train_topic(service):
    """
    Tests whether the control channel listener activates the train mode of the service if it
    receives the service train topic.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    service._register_with_dialogsystem()
    service.is_training = False
    expected_topics = [service._train_topic, f"ACK/{service._train_topic}"]

    q = get_all_messages_to_published_topic(topics=[service._train_topic],
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)
    assert service.is_training is True


@pytest.mark.terminate_topics('_terminate_topic')
def test_control_channel_listener_for_eval_topic(service):
    """
    Tests whether the control channel listener activates the evaluation mode of the service if it
    receives the service evaluation topic.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    service._register_with_dialogsystem()
    service.is_training = True
    expected_topics = [service._eval_topic, f"ACK/{service._eval_topic}"]

    q = get_all_messages_to_published_topic(topics=[service._eval_topic],
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)
    assert service.is_training is False


@pytest.mark.terminate_topics('_terminate_topic')
def test_control_channel_listener_for_unknown_topic(service):
    """
    Tests whether the control channel listener does not send any messages to internal subscribers
    if it receives an unknown topic.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    service._register_with_dialogsystem()
    unknown_topic = 'foo'

    q = get_all_messages_to_published_topic(topics=[unknown_topic],
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == {unknown_topic}


def test_train():
    """
    Tests whether calling the train method activates the train mode of the service.
    """
    service = Service()
    service.is_training = False
    service.train()
    assert service.is_training is True


def test_eval():
    """
    Tests whether calling the evaluation method activates the evaluation mode of the service.
    """
    service = Service()
    service.is_training = True
    service.eval()
    assert service.is_training is False


@pytest.mark.terminate_topics('_terminate_topic')
def test_run_standalone(service):
    """
    Tests whether running a service standalone as a remote service registers the system as such.
    Note: This test simulates a dialog system to communicate with the service.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._identifier = 'remote_service'
    thread = threading.Thread(target=service.run_standalone)
    thread.start()

    ctx = zmq.Context.instance()
    rep_socket = ctx.socket(zmq.REP)
    rep_socket.bind(f'{service._protocol}://{service._host_addr}:65535')

    messages = []
    remote_service_identifier = None
    reg_data = None

    while len(messages) < 2:
        msg, data = rep_socket.recv_multipart()
        msg = msg.decode("utf-8")
        messages.append(msg)
        if msg.startswith("REGISTER_"):
            remote_service_identifier = msg[len("REGISTER_"):]
            reg_data = pickle.loads(data)
            rep_socket.send(
                    bytes(f'ACK_REGISTER_{remote_service_identifier}', encoding="ascii"))

    thread.join(timeout=5)

    assert len(messages) == 2
    assert messages[0] == f'REGISTER_{service._identifier}'
    assert messages[1] == f'CONF_REGISTER_{service._identifier}'
    assert remote_service_identifier == service._identifier
    assert reg_data is not None
    assert service._domain_name in reg_data
    assert service._sub_topics in reg_data
    assert service._pub_topics in reg_data
    assert service._start_topic in reg_data
    assert service._end_topic in reg_data
    assert service._terminate_topic in reg_data


def test_get_all_subscribed_topics():
    """
    Tests whether requesting all subscribed topics actually returns all subscribed topics.
    """
    service = Service()
    sub_topics = ['foo', 'bar']
    service._sub_topics = sub_topics
    res = service.get_all_subscribed_topics()
    assert res == sub_topics


def test_get_all_published_topics():
    """
    Tests whether requesting all published topics actually returns all published topics.
    """
    service = Service()
    pub_topics = ['foo', 'bar']
    service._pub_topics = pub_topics
    res = service.get_all_published_topics()
    assert res == pub_topics


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_receiver_thread_for_start_topic(service):
    """
    Tests whether a subscriber acknowledges its start topic when it receives it.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    expected_topics = list(service._internal_start_topics.keys())
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]

    q = get_all_messages_to_published_topic(topics=service._internal_start_topics,
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_receiver_thread_for_end_topic(service):
    """
    Tests whether a subscriber acknowledges its end topic when it receives it.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    expected_topics = list(service._internal_end_topics.keys())
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]

    q = get_all_messages_to_published_topic(topics=service._internal_end_topics,
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)


def test_receiver_thread_for_terminate_topic(service):
    """
    Tests whether a subscriber acknowledges its terminate topic when it receives it.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    expected_topics = list(service._internal_terminate_topics.keys())
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]

    q = get_all_messages_to_published_topic(topics=service._internal_terminate_topics,
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_receiver_thread_for_known_topics(service):
    """
    Tests whether a subscriber executes the function it subscribes for when it receives
    messages for all of the function's subscribed topics.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    service.dummy_function_result = None
    sub_topics = [(topic, i) for i, topic in enumerate(service.dummy_function.sub_topics)]
    queued_sub_topics = [(topic, (i+1)*10) for i, topic in enumerate(
            service.dummy_function.queued_sub_topics)]
    start_topics = [(topic, True) for topic in service._internal_start_topics.keys()]

    q = get_all_messages_to_published_topic(topics=start_topics + sub_topics + queued_sub_topics,
                                            protocol=service._protocol, host=service._host_addr,
                                            pub_port=service._pub_port, sub_port=service._sub_port)

    result = service.dummy_function_result
    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == len(sub_topics + queued_sub_topics)
    assert all(topic in result.keys() for topic, _ in sub_topics + queued_sub_topics)
    assert all(result[topic] == content for topic, content in sub_topics)
    assert all(content in result[topic] for topic, content in queued_sub_topics)
    assert set(recv_topics) >= set(service.dummy_function.pub_topics)


@pytest.mark.terminate_topics('_internal_terminate_topics')
def test_receiver_thread_for_incomplete_topic_list(service):
    """
    Tests whether a subscriber does not execute the function it subscribes for when it does not
    receive messages for all of the function's subscribed topics but only for some of them.

    Args:
        service (Service): Instance of a class that inherits of the Service class.
    """
    service._init_pubsub()
    service.dummy_function_result = None
    sub_topics = [(topic, i) for i, topic in enumerate(service.dummy_function.sub_topics)]
    queued_sub_topics = []
    start_topics = [(topic, True) for topic in service._internal_start_topics.keys()]

    get_all_messages_to_published_topic(topics=start_topics + sub_topics + queued_sub_topics,
                                        protocol=service._protocol, host=service._host_addr,
                                        pub_port=service._pub_port, sub_port=service._sub_port)

    result = service.dummy_function_result
    assert result is None
