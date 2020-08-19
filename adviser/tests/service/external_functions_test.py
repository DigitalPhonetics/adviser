import os
import sys
import pickle
from time import sleep
import zmq
from zmq.devices import ProcessProxy
from multiprocessing import Process


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(get_root_dir())
from services.service import _send_msg, _send_ack, _recv_ack


# auxiliary functions

def create_proxy():
    """
    Creates a proxy for communication between publisher and subscriber.
    """
    proxy = ProcessProxy(in_type=zmq.XSUB, out_type=zmq.XPUB)
    proxy.bind_in(f"tcp://127.0.0.1:65534")
    proxy.bind_out(f"tcp://127.0.0.1:65533")
    proxy.start()
    return proxy


def create_publisher():
    """
    Creates a publisher socket.
    """
    ctx = zmq.Context.instance()
    publisher = ctx.socket(zmq.PUB)
    publisher.sndhwm = 1100000
    publisher.connect(f"tcp://127.0.0.1:65534")
    return publisher


def create_subscriber():
    """
    Creates a subscriber socket.
    """
    ctx = zmq.Context.instance()
    subscriber = ctx.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')  # subscribe to all topics
    subscriber.connect(f"tcp://127.0.0.1:65533")
    sleep(.25)
    return subscriber


# Test functions

def test_send_msg():
    """
    Tests whether sending a message delivers the message correctly.
    """
    create_proxy()
    publisher = create_publisher()
    subscriber = create_subscriber()

    topic = 'foo'
    content = 'bar'

    _send_msg(publisher, topic, content)
    msg = subscriber.recv_multipart()

    publisher.close()
    subscriber.close()

    assert msg is not None
    assert len(msg) == 2
    assert msg[0].decode('ascii') == topic
    msg_content = pickle.loads(msg[1])
    assert len(msg_content) == 2
    assert isinstance(msg_content[0], float)
    assert msg_content[1] == content


def test_send_ack():
    """
    Tests whether sending an acknowledgement delivers the message correctly.
    """
    create_proxy()
    publisher = create_publisher()
    subscriber = create_subscriber()

    topic = 'foo'
    content = True

    _send_ack(publisher, topic, content)
    msg = subscriber.recv_multipart()

    publisher.close()
    subscriber.close()

    assert msg is not None
    assert len(msg) == 2
    msg_topic = msg[0].decode('ascii')
    assert '/' in msg_topic
    assert msg_topic.split('/')[0] == 'ACK'
    assert msg_topic.split('/')[1] == topic
    msg_content = pickle.loads(msg[1])
    assert len(msg_content) == 2
    assert isinstance(msg_content[0], float)
    assert msg_content[1] == content


def test_recv_ack_for_expected_content():
    """
    Tests whether receiving an acknowledgement is successful if the content of the message is as
    expected. To measure the success here, it is checked whether the process for receiving the
    acknowledgement terminates or not.
    """
    create_proxy()
    publisher = create_publisher()

    topic = 'foo'
    content = True
    expected_content = True

    def recv_ack(topic, expected_content):
        subscriber = create_subscriber()
        _recv_ack(subscriber, topic, expected_content)
        subscriber.close()

    process = Process(target=recv_ack, args=(topic, expected_content))
    process.start()
    sleep(.5)
    _send_ack(pub_channel=publisher, topic=topic, content=content)
    process.join(timeout=.1)  # kills process if recv_ack ended
    process_is_alive = process.is_alive()
    process.terminate()    # kills process in any case
    publisher.close()
    assert process_is_alive is False


def test_recv_ack_for_unexpected_content():
    """
    Tests whether receiving an acknowledgement is not successful if the content of the message
    is not as expected. To measure the success here, it is checked whether the process for
    receiving the acknowledgement terminates or not.
    """
    create_proxy()
    publisher = create_publisher()

    topic = 'foo'
    content = True
    expected_content = False

    def recv_ack(topic, expected_content):
        subscriber = create_subscriber()
        _recv_ack(subscriber, topic, expected_content)
        subscriber.close()

    process = Process(target=recv_ack, args=(topic, expected_content))
    process.start()
    sleep(.5)
    _send_ack(pub_channel=publisher, topic=topic, content=content)
    process.join(timeout=.1)  # kills process if recv_ack ended
    process_is_alive = process.is_alive()
    process.terminate()  # kills process in any case
    publisher.close()
    assert process_is_alive is True
