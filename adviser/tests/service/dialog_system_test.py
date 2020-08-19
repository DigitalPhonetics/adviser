import os
import sys
import zmq
from time import sleep
import threading
from queue import Queue
import pickle


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.service import DialogSystem, _send_msg, RemoteService
from utils.topics import Topic


# auxiliary function
def recv_all_messages(q, protocol,sub_port):
    """
    Receives all messages that are sent to any topics.

    Args:
        q (Queue): Queue to store the received messages in.
        protocol (str): Communication protocol.
        sub_port (int): Subscriber port.
    """
    ctx = zmq.Context.instance()
    subscriber = ctx.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')  # subscribe to all topics
    subscriber.connect(f"{protocol}://127.0.0.1:{sub_port}")
    sleep(.1)
    while True:
        msg = subscriber.recv_multipart(copy=True)
        if msg[0].decode('ascii') == 'stop_test':
            break
        q.put(msg)

    subscriber.close()


# Test functions

def test_init_registers_services(serviceA, serviceB, serviceC):
    """
    Tests whether initializing a dialog system will initialize the publish-subscribe network for
    all of its services and will register these services.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    ds = DialogSystem(services=[serviceA, serviceB, serviceC])
    assert ds._sub_topics != dict()
    assert ds._pub_topics != dict()
    assert ds._start_topics != set()
    assert ds._end_topics != set()
    assert ds._terminate_topics != set()
    ds.shutdown()


def test_register_single_pub_topic():
    """
    Tests whether registering a topic to a publisher will store this topic and associate
    the publisher with the topic.
    """
    ds = DialogSystem(services=[])
    publisher = 'foo'
    topic = 'bar'
    ds._pub_topics = dict()
    ds._register_pub_topic(publisher, topic)
    assert ds._pub_topics != dict()
    assert isinstance(ds._pub_topics, dict)
    assert len(ds._pub_topics) == 1
    assert topic in ds._pub_topics
    assert isinstance(ds._pub_topics[topic], set)
    assert publisher in ds._pub_topics[topic]
    ds.shutdown()


def test_register_multiple_pub_topics():
    """
    Tests whether registering a topic to multiple publishers will store this topic and associate
    both publishers with the topic.
    """
    ds = DialogSystem(services=[])
    publisher1 = 'foo'
    publisher2 = 'baz'
    topic = 'bar'
    ds._pub_topics = dict()
    ds._register_pub_topic(publisher1, topic)
    ds._register_pub_topic(publisher2, topic)
    assert ds._pub_topics != dict()
    assert isinstance(ds._pub_topics, dict)
    assert len(ds._pub_topics) == 1
    assert topic in ds._pub_topics
    assert isinstance(ds._pub_topics[topic], set)
    assert publisher1 in ds._pub_topics[topic]
    assert publisher2 in ds._pub_topics[topic]
    ds.shutdown()


def test_register_single_sub_topic():
    """
    Tests whether registering a topic to a subscriber will store this topic and associate
    the subscriber with the topic.
    """
    ds = DialogSystem(services=[])
    subscriber = 'foo'
    topic = 'bar'
    ds._sub_topics = dict()
    ds._register_sub_topic(subscriber, topic)
    assert ds._sub_topics != dict()
    assert isinstance(ds._sub_topics, dict)
    assert len(ds._sub_topics) == 1
    assert topic in ds._sub_topics
    assert isinstance(ds._sub_topics[topic], set)
    assert subscriber in ds._sub_topics[topic]
    ds.shutdown()


def test_register_multiple_sub_topics():
    """
    Tests whether registering a topic to multiple subscribers will store this topic and associate
    both subscribers with the topic.
    """
    ds = DialogSystem(services=[])
    subscriber1 = 'foo'
    subscriber2 = 'baz'
    topic = 'bar'
    ds._sub_topics = dict()
    ds._register_sub_topic(subscriber1, topic)
    ds._register_sub_topic(subscriber2, topic)
    assert ds._sub_topics != dict()
    assert isinstance(ds._sub_topics, dict)
    assert len(ds._sub_topics) == 1
    assert topic in ds._sub_topics
    assert isinstance(ds._sub_topics[topic], set)
    assert subscriber1 in ds._sub_topics[topic]
    assert subscriber2 in ds._sub_topics[topic]
    ds.shutdown()


def test_register_remote_services_without_remote_services():
    """
    Tests whether registering a remote service without giving any remote services will not
    register anything.
    """
    ds = DialogSystem(services=[])
    ds._register_remote_services(remote_services=[], reg_port=65535)
    assert ds._sub_topics == dict()
    assert ds._pub_topics == dict()
    assert ds._start_topics == set()
    assert ds._end_topics == set()
    assert ds._terminate_topics == set()
    assert ds._remote_identifiers == set()
    ds.shutdown()


def test_register_remote_services(serviceB):
    """
    Tests whether registering a remote service will register the service correctly.

    Args:
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
    """
    ds = DialogSystem(services=[])
    identifier = 'remote_service'
    serviceB._identifier = identifier

    thread = threading.Thread(target=serviceB.run_standalone)
    thread.start()
    sleep(.25)

    remote_service = RemoteService(identifier=identifier)

    ds._register_remote_services(remote_services={identifier: remote_service}, reg_port=65535)
    thread.join(5)
    sleep(.1)

    assert serviceB._identifier in ds._remote_identifiers
    assert serviceB.domain in ds._domains
    assert all(topic in ds._sub_topics for topic in serviceB._sub_topics)
    assert all(topic in ds._pub_topics for topic in serviceB._pub_topics)
    assert serviceB._start_topic in ds._start_topics
    assert serviceB._end_topic in ds._end_topics
    assert serviceB._terminate_topic in ds._terminate_topics
    ds.shutdown()


def test_add_service_info_for_sub_topics():
    """
    Tests whether adding the service information for a service with subscribed topics will store
    these topics as well as the start, end and terminate topics of the service.
    """
    ds = DialogSystem(services=[])
    service_name = 'foo'
    domain_name = 'bar'
    sub_topics = ['one', 'two']
    pub_topics = []
    start_topic = 'foo_start'
    end_topic = 'foo_end'
    terminate_topic = 'foo_terminate'
    ds._add_service_info(service_name=service_name, domain_name=domain_name,
                         sub_topics=sub_topics, pub_topics=pub_topics, start_topic=start_topic,
                         end_topic=end_topic, terminate_topic=terminate_topic)
    assert domain_name in ds._domains
    assert all(topic in ds._sub_topics for topic in sub_topics)
    assert all(ds._sub_topics[topic] == {service_name} for topic in sub_topics)
    assert ds._pub_topics == dict()
    assert start_topic in ds._start_topics
    assert end_topic in ds._end_topics
    assert terminate_topic in ds._terminate_topics


def test_add_service_info_for_pub_topics():
    """
    Tests whether adding the service information for a service with published topics will store
    these topics as well as the start, end and terminate topics of the service.
    """
    ds = DialogSystem(services=[])
    service_name = 'foo'
    domain_name = 'bar'
    sub_topics = []
    pub_topics = ['one', 'two']
    start_topic = 'foo_start'
    end_topic = 'foo_end'
    terminate_topic = 'foo_terminate'
    ds._add_service_info(service_name=service_name, domain_name=domain_name,
                         sub_topics=sub_topics, pub_topics=pub_topics, start_topic=start_topic,
                         end_topic=end_topic, terminate_topic=terminate_topic)
    print(ds._pub_topics)
    assert domain_name in ds._domains
    assert ds._sub_topics == dict()
    assert all(topic in ds._pub_topics for topic in pub_topics)
    assert all(ds._pub_topics[topic] == {service_name} for topic in pub_topics)
    assert start_topic in ds._start_topics
    assert end_topic in ds._end_topics
    assert terminate_topic in ds._terminate_topics


def test_setup_dialog_end_listener():
    """
    Tests whether setting up a dialog end listener will add another socket.
    """
    ds = DialogSystem(services=[])
    ds._setup_dialog_end_listener()
    assert hasattr(ds, '_end_socket')
    assert isinstance(ds._end_socket, zmq.Socket)
    ds.shutdown()


def test_stop():
    """
    Tests whether stopping the dialog system will set the stop variable to True.
    """
    ds = DialogSystem(services=[])
    ds._stopEvent.clear()
    ds.stop()
    assert ds._stopEvent.is_set()
    ds.shutdown()


def test_terminating():
    """
    Tests whether terminating returns True if the dialog system has been stopped before.
    """
    ds = DialogSystem(services=[])
    ds._stopEvent.set()
    res = ds.terminating()
    assert res is True
    ds.shutdown()


def test_shutdown(serviceA, serviceB, serviceC):
    """
    Tests whether performing a shutdown on the dialog system terminates all services of the system.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    services = [serviceA, serviceB, serviceC]
    ds = DialogSystem(services=services)
    expected_topics = [serviceA._terminate_topic, *serviceA._internal_terminate_topics.keys(),
                       serviceB._terminate_topic, *serviceB._internal_terminate_topics.keys(),
                       serviceC._terminate_topic, *serviceC._internal_terminate_topics.keys()]
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]

    q = Queue()
    thread = threading.Thread(target=recv_all_messages, args=(q, ds.protocol, ds._sub_port))
    thread.start()
    sleep(.25)
    ds.shutdown()
    sleep(.5)
    _send_msg(ds._control_channel_pub, 'stop_test', True)
    thread.join(5)

    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)


def test_end_dialog(serviceA, serviceB, serviceC):
    """
    Tests whether ending a dialog will send end messages to all services of the dialog system.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    services = [serviceA, serviceB, serviceC]
    ds = DialogSystem(services=services)
    expected_topics = [serviceA._end_topic, *serviceA._internal_end_topics.keys(),
                       serviceB._end_topic, *serviceB._internal_end_topics.keys(),
                       serviceC._end_topic, *serviceC._internal_end_topics.keys()]
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]
    expected_topics.append(Topic.DIALOG_END)

    ctx = zmq.Context.instance()
    publisher = ctx.socket(zmq.PUB)
    publisher.sndhwm = 1100000
    publisher.connect(f"{ds.protocol}://127.0.0.1:{ds._pub_port}")

    q = Queue()
    recv_thread = threading.Thread(target=recv_all_messages, args=(q, ds.protocol, ds._sub_port))
    recv_thread.start()
    end_thread = threading.Thread(target=ds._end_dialog)
    end_thread.start()
    sleep(.25)

    _send_msg(publisher, Topic.DIALOG_END, True)
    sleep(.5)
    _send_msg(publisher, 'stop_test', True)
    recv_thread.join(5)
    end_thread.join(5)

    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)
    ds.shutdown()


def test_start_dialog_for_start_topics(serviceA, serviceB, serviceC):
    """
    Tests whether starting a dialog will send start messages to all services in the dialog system.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    services = [serviceA, serviceB, serviceC]
    ds = DialogSystem(services=services)
    expected_topics = [serviceA._start_topic, *serviceA._internal_start_topics.keys(),
                       serviceB._start_topic, *serviceB._internal_start_topics.keys(),
                       serviceC._start_topic, *serviceC._internal_start_topics.keys()]
    expected_topics += [f"ACK/{topic}" for topic in expected_topics]

    q = Queue()
    thread = threading.Thread(target=recv_all_messages, args=(q, ds.protocol, ds._sub_port))
    thread.start()
    sleep(.25)
    ds._start_dialog(start_signals={})
    sleep(.5)
    _send_msg(ds._control_channel_pub, 'stop_test', True)
    thread.join(5)

    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    assert q.empty() is False
    assert set(recv_topics) == set(expected_topics)
    ds.shutdown()


def test_start_dialog_for_start_signals(serviceA, serviceB, serviceC):
    """
    Tests whether starting a dialog will send start signal messages if any start signals are given.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    services = [serviceA, serviceB, serviceC]
    ds = DialogSystem(services=services)
    start_signals = {'foo': 'foo_start', 'bar': 'bar_start'}

    q = Queue()
    thread = threading.Thread(target=recv_all_messages, args=(q, ds.protocol, ds._sub_port))
    thread.start()
    sleep(.25)
    ds._start_dialog(start_signals=start_signals)
    sleep(.5)
    _send_msg(ds._control_channel_pub, 'stop_test', True)
    thread.join(5)

    recv_messages = {msg[0].decode('ascii'): pickle.loads(msg[1])[1] for msg in q.queue}
    assert q.empty() is False
    assert len(recv_messages) >= len(start_signals)
    assert all(topic in recv_messages for topic in start_signals.keys())
    assert all(recv_messages[topic] == content for topic, content in start_signals.items())
    ds.shutdown()


def test_run_dialog(serviceA, serviceB, serviceC):
    """
    Tests whether running a dialog first starts the dialog and finally ends it.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    ds = DialogSystem(services=[serviceA, serviceB, serviceC])
    start_signals = {}

    ctx = zmq.Context.instance()
    publisher = ctx.socket(zmq.PUB)
    publisher.sndhwm = 1100000
    publisher.connect(f"{ds.protocol}://127.0.0.1:{ds._pub_port}")

    q = Queue()
    recv_thread = threading.Thread(target=recv_all_messages, args=(q, ds.protocol, ds._sub_port))
    recv_thread.start()
    sleep(.25)
    dialog_thread = threading.Thread(target=ds.run_dialog, args=(start_signals,))
    dialog_thread.start()
    sleep(.5)

    _send_msg(publisher, Topic.DIALOG_END, True)
    sleep(.5)
    _send_msg(publisher, 'stop_test', True)
    recv_thread.join(5)
    dialog_thread.join(5)

    recv_topics = [msg[0].decode('ascii') for msg in q.queue]
    start_topics = [topic for topic in recv_topics if '/START' in topic]
    end_topics = [topic for topic in recv_topics if '/END' in topic]
    assert q.empty() is False
    assert len(start_topics) == len(end_topics)
    assert all(topic.replace('/START', '/END') in end_topics for topic in start_topics)
    ds.shutdown()


def test_list_published_topics(serviceA, serviceB, serviceC):
    """
    Tests whether listing the published topics will return all published topics.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    ds = DialogSystem(services=[serviceA, serviceB, serviceC])
    res = ds.list_published_topics()
    assert res == ds._pub_topics
    ds.shutdown()


def test_list_subscribed_topics(serviceA, serviceB, serviceC):
    """
    Tests whether listing the subscribed topics will return all subscribed topics.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    ds = DialogSystem(services=[serviceA, serviceB, serviceC])
    res = ds.list_subscribed_topics()
    assert res == ds._sub_topics
    ds.shutdown()


def test_list_inconsistencies_in_sub_topics(serviceA, serviceB, serviceC_with_inconsistent_subscriber):
    """
    Tests whether listing inconsistencies will detect these inconsistencies if they occur in the
    subscribed topics.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC_with_inconsistent_subscriber (ServiceC): Instance of the ServiceC class which
        inherits from Service and which leads to an inconsistency on the subscriber side when
        paired with serviceA and serviceB (given in conftest_services.py)
    """
    ds = DialogSystem(services=[serviceA, serviceB, serviceC_with_inconsistent_subscriber])
    errors, warnings = ds.list_inconsistencies()
    assert errors != dict()
    assert len(errors) > 0
    assert all(topic in ds._sub_topics.keys() for topic in errors.keys())
    assert all(value in ds._sub_topics.values() for value in errors.values())
    assert warnings == dict()
    ds.shutdown()


def test_list_inconsistencies_in_pub_topics(serviceA, serviceB,
                                            serviceC_with_inconsistent_publisher):
    """
    Tests whether listing inconsistencies will detect these inconsistencies if they occur in the
    published topics.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC_with_inconsistent_publisher (ServiceC): Instance of the ServiceC class which
        inherits from Service and which leads to an inconsistency on the publisher side when
        paired with serviceA and serviceB (given in conftest_services.py)
    """
    ds = DialogSystem(services=[serviceA, serviceB, serviceC_with_inconsistent_publisher])
    errors, warnings = ds.list_inconsistencies()
    assert errors == dict()
    assert warnings != dict()
    assert len(warnings) > 0
    assert all(topic in ds._pub_topics.keys() for topic in warnings.keys())
    assert all(value in ds._pub_topics.values() for value in warnings.values())
    ds.shutdown()


def test_is_error_free_messaging_pipeline_with_inconsistencies(serviceA, serviceB, serviceC_with_inconsistent_subscriber):
    """
    Tests whether the check for an error free messaging pipeline fails if there are any
    inconsistencies.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC_with_inconsistent_subscriber (ServiceC): Instance of the ServiceC class which
        inherits from Service and which leads to an inconsistency on the subscriber side when
        paired with serviceA and serviceB (given in conftest_services.py)
    """
    ds = DialogSystem(services=[serviceA, serviceB, serviceC_with_inconsistent_subscriber])
    res = ds.is_error_free_messaging_pipeline()
    assert res is False
    ds.shutdown()


def test_is_error_free_messaging_pipeline_without_inconsistencies(serviceA, serviceB, serviceC):
    """
    Tests whether the check for an error free messaging pipeline succeeds if there are no
    inconsistencies.

    Args:
        serviceA (ServiceA): Instance of the ServiceA class which inherits from Service (given in
        conftest_services.py)
        serviceB (ServiceB): Instance of the ServiceB class which inherits from Service (given in
        conftest_services.py)
        serviceC (ServiceC): Instance of the ServiceC class which inherits from Service (given in
        conftest_services.py)
    """
    ds = DialogSystem(services=[serviceA, serviceB, serviceC])
    res = ds.is_error_free_messaging_pipeline()
    assert res is True
    ds.shutdown()