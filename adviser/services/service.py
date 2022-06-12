############################################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify'
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
############################################################################################

import copy
import datetime
import inspect
import pickle
import threading
import time
from threading import Thread
from typing import List, Dict, Union, Iterable, Any
import platform

import zmq
from zmq import Context, Socket
from zmq.devices import ThreadProxy, ProcessProxy

from utils.domain.domain import Domain
from utils.logger import DiasysLogger
from utils.topics import Topic


def _send_msg(pub_channel: Socket, topic: str, content: Any):
    """ Serializes message, appends current timespamp and sends it over the specified channel to the specified topic.
        Use this function for all internal message passing.

    Args:
        pub_channel (Socket): publisher socket
        topic (str): topic to publish to
        content (Any): message content
     """
    timestamp = datetime.datetime.now().timestamp()  # current timestamp as POSIX float
    data = pickle.dumps((timestamp, content))
    pub_channel.send_multipart((bytes(topic, encoding="ascii"), data))


def _send_ack(pub_channel: Socket, topic: str, content: bool = True):
    """ Sends an acknowledge-message to the specified channel (ACK).
        Is used together with `_recv_ack` to synchronize services (waiting for ACK messages).
    
    Args:
        pub_channel (Socket): publisher socket
        topic (str): topic to send ACK to
        content (bool): for ACK's, content is either `True` (ACK) or `False` (NACK)
    """
    _send_msg(pub_channel, f"ACK/{topic}", content)


def _recv_ack(sub_channel: Socket, topic: str, expected_content: bool = True):
    """ Blocks until an acknowledge-message for the specified topic with the expected content is received via the
        specified subscriber channel. 
    
    Args:
        sub_channel (Socket): subscriber socket
        topic (str): topic to listen for ACK's
        expected_content (bool): are we expecting `True` (ACK) or `False` (NACK)
    """
    ack_topic = topic if topic.startswith("ACK/") else f"ACK/{topic}"
    while True:
        msg = sub_channel.recv_multipart(copy=True)
        recv_topic = msg[0].decode("ascii")
        content = pickle.loads(msg[1])[1]  # pickle.loads(msg) -> tuple(timestamp, content) -> return content
        if recv_topic == ack_topic:
            if content == expected_content:
                return


class RemoteService:
    """
    This is a placeholder` to be used in the service list argument when constructing a `DialogSystem`:
    * Run the real `Service` instance on a remote node, give it a *UNIQUE* identifier
        * call `run_standalone()` on this instance
    * Instantiate a remote service on the node about to run the `DialogSystem`, assign the *SAME* identifier to it
        * add it to the `DialogSystem` service list
    * Now, when calling the constructor of `DialogSystem`, you should see messages informing you about the 
      successfull connection, or if the system is still trying to connect, it will block until connected to
      the remote service.
    """
    def __init__(self, identifier: str):
        """
        Args:
            identifier (str): the *UNIQUE* identifier to call the remote service instance
        """
        self.identifier = identifier


class Service:
    """
    Service base class.
    Inherit from this class, if you want to publish / subscribe to topics *(Don't forget to call the super constructor!)*.
    You may decorate arbitrary functions in the child class with the services.service.PublishSubscribe decorator
    for this purpose.

    Note: A `Service` will only start listening to messages once it is added to a `DialogSystem` 
          (or calling `run_standalone()` in the remote case and adding a corresponding `RemoteService` to the `DialogSystem`).
    """

    def __init__(self, domain: Union[str, Domain] = "", sub_topic_domains: Dict[str, str] = {}, pub_topic_domains: Dict[str, str] = {},
                 ds_host_addr: str = "127.0.0.1", sub_port: int = 65533, pub_port: int = 65534, protocol: str = "tcp",
                 debug_logger: DiasysLogger = None, identifier: str = None):
        """
        Create a new service instance *(call this super constructor from your inheriting classes!)*.
        
        Args:
            domain (Union[str, Domain]): The domain(-name) of your service (or empty string, if domain-agnostic).
                                         If a domain(-name) is set, it will automatically filter out all messages from other domains.
                                         If no domain(-name) is set, messages from all domains will be received.
            sub_topic_domains (Dict[str, str]): change subscribed to topics to listen to a specific domain 
                                                (e.g. 'erase'/append a domain for a specific topic)
            pub_topic_domains (Dict[str, str]): change published topics to a specific domain
                                               (e.g. 'erase'/append a domain for a specific topic)
            ds_host_addr (str): IP-address of the parent `DialogSystem` (default: localhost)
            sub_port (int): subscriber port following zmq's XSUB/XPUB pattern
            pub_port (int): publisher port following zmq's XSUB/XPUB pattern
            protocol (string): communication protocol with `DialogSystem` - has to match!
                               Possible options: `tcp`, `inproc`, `ipc`
            debug_logger (DiasysLogger): If not `None`, all messags are printed to the logger, including send/receive events.
                                         Can be useful for debugging because you can still see messages received by the `DialogSystem`
                                         even if they are never forwarded (as expected) to your `Service`.
            identifier (str): Set this to a *UNIQUE* identifier per service to be run remotely.
                              See `RemoteService` for more details.
        """

        self.is_training = False
        self.domain = domain
        # get domain name (gets appended to all sub/pub topics so that different domain topics don't get shared)
        if domain is not None:
            self._domain_name = domain.get_domain_name() if isinstance(domain, Domain) else domain
        else:
            self._domain_name = ""
        self._sub_topic_domains = sub_topic_domains
        self._pub_topic_domains = pub_topic_domains

        # socket information
        self._host_addr = ds_host_addr
        self._sub_port = sub_port
        self._pub_port = pub_port
        self._protocol = protocol
        self._identifier = identifier

        self.debug_logger = debug_logger

        self._sub_topics = set()
        self._pub_topics = set()
        self._publish_sockets = dict()

        self._internal_start_topics = dict()
        self._internal_end_topics = dict()
        self._internal_terminate_topics = dict()

        # NOTE: class name + memory pointer make topic unique (required, e.g. for running mutliple instances of same module!)
        self._start_topic = f"{type(self).__name__}/{id(self)}/START"
        self._end_topic = f"{type(self).__name__}/{id(self)}/END"
        self._terminate_topic = f"{type(self).__name__}/{id(self)}/TERMINATE"
        self._train_topic = f"{type(self).__name__}/{id(self)}/TRAIN"
        self._eval_topic = f"{type(self).__name__}/{id(self)}/EVAL"

    def _init_pubsub(self): 
        """ Search for all functions decorated with the `PublishSubscribe` decorator and call the setup methods for them """
        for func_name in dir(self):
            func_inst = getattr(self, func_name)
            if hasattr(func_inst, "pubsub"):
                # found decorated publisher / subscriber function -> setup sockets and listeners
                self._setup_listener(func_inst, getattr(func_inst, "sub_topics"),
                                     getattr(func_inst, 'queued_sub_topics'))
                self._setup_publishers(func_inst, getattr(func_inst, "pub_topics"))

    def _register_with_dialogsystem(self):
        """ Start listening to dialog system control channel messages """
        self._setup_dialog_ctrl_msg_listener()
        Thread(target=self._control_channel_listener).start()

    def _setup_listener(self, func_instance, topics: List[str], queued_topics: List[str]):
        """
        Starts a new subscription thread for a function decorated with `services.service.PublishSubscribe`.
        
        Args:
            func_instance (function): instance of the function that was decorated with `services.service.PublishSubscribe`.
            topics (List[str]): list of subscribed topics (drops all but most recent messages before function call)
            queued_topics (List[str]): list for subscribed topics (drops no messages, forward a list of received messages to function call)
        """
        if len(topics + queued_topics) == 0:
            # no subscribed to topics - no need to setup anything (e.g. only publisher)
            return
            # ensure that sub_topics and queued_sub_topics don't intersect (otherwise, both would set same function argument value)
        assert set(topics).isdisjoint(queued_topics), "sub_topics and queued_sub_topics have to be disjoint!"

        # setup socket
        ctx = Context.instance()
        subscriber = ctx.socket(zmq.SUB)
        # subscribe to all listed topics
        for topic in topics + queued_topics:
            topic_domain_str = f"{topic}/{self._domain_name}" if self._domain_name else topic
            if topic in self._sub_topic_domains:
                # overwrite domain for this specific topic and service instance
                topic_domain_str = f"{topic}/{self._sub_topic_domains[topic]}" if self._sub_topic_domains[topic] else topic
            subscriber.setsockopt(zmq.SUBSCRIBE, bytes(topic_domain_str, encoding="ascii"))
        # subscribe to control channels
        subscriber.setsockopt(zmq.SUBSCRIBE, bytes(f"{func_instance}/START", encoding="ascii"))
        subscriber.setsockopt(zmq.SUBSCRIBE, bytes(f"{func_instance}/END", encoding="ascii"))
        subscriber.setsockopt(zmq.SUBSCRIBE, bytes(f"{func_instance}/TERMINATE", encoding="ascii"))
        subscriber.connect(f"{self._protocol}://{self._host_addr}:{self._sub_port}")
        self._internal_start_topics[f"{str(func_instance)}/START"] = str(func_instance)
        self._internal_end_topics[f"{str(func_instance)}/END"] = str(func_instance)
        self._internal_terminate_topics[f"{str(func_instance)}/TERMINATE"] = str(func_instance)

        # register and run listener thread
        listener_thread = Thread(target=self._receiver_thread, args=(subscriber, func_instance,
                                                                     topics, queued_topics,
                                                                     f"{str(func_instance)}/START",
                                                                     f"{str(func_instance)}/END",
                                                                     f"{str(func_instance)}/TERMINATE"))
        listener_thread.start()

        # add to list of local topics
        # TODO maybe add topic_domain_str instead for more clarity?
        self._sub_topics.update(topics + queued_topics)

    def _setup_publishers(self, func_instance, topics):
        """ Creates a publish socket for a function decorated with `services.service.PublishSubscribe`. """
        if len(topics) == 0:
            return # no topics - no need for a socket

        # setup publish socket
        ctx = Context.instance()
        publisher = ctx.socket(zmq.PUB)
        publisher.sndhwm = 1100000
        publisher.connect(f"{self._protocol}://{self._host_addr}:{self._pub_port}")
        self._publish_sockets[func_instance] = publisher

        # add to list of local topics
        self._pub_topics.update(topics)

    def _setup_dialog_ctrl_msg_listener(self):
        """ Setup a subscriber socket to receive `DialogSystem` control message """ 
         
        ctx = Context.instance()

        # setup receiver for dialog system control messages
        self._control_channel_sub = ctx.socket(zmq.SUB)
        self._control_channel_sub.setsockopt(zmq.SUBSCRIBE, bytes(self._start_topic, encoding="ascii"))
        self._control_channel_sub.setsockopt(zmq.SUBSCRIBE, bytes(self._end_topic, encoding="ascii"))
        self._control_channel_sub.setsockopt(zmq.SUBSCRIBE, bytes(self._terminate_topic, encoding="ascii"))
        self._control_channel_sub.setsockopt(zmq.SUBSCRIBE, bytes(self._train_topic, encoding="ascii"))
        self._control_channel_sub.setsockopt(zmq.SUBSCRIBE, bytes(self._eval_topic, encoding="ascii"))
        self._control_channel_sub.connect(f"{self._protocol}://{self._host_addr}:{self._sub_port}")

        # setup sender for dialog system control message acknowledgements 
        self._control_channel_pub = ctx.socket(zmq.PUB)
        self._control_channel_pub.sndhwm = 1100000
        self._control_channel_pub.connect(f"{self._protocol}://{self._host_addr}:{self._pub_port}")

        # setup receiver for internal ACK messages
        self._internal_control_channel_sub = ctx.socket(zmq.SUB)
        for internal_ctrl_topic in list(self._internal_end_topics.keys()) + list(
                self._internal_start_topics.keys()) + list(self._internal_terminate_topics.keys()):
            self._internal_control_channel_sub.setsockopt(zmq.SUBSCRIBE,
                                                          bytes(f"ACK/{internal_ctrl_topic}", encoding="ascii"))
        self._internal_control_channel_sub.connect(f"{self._protocol}://{self._host_addr}:{self._sub_port}")

    def _control_channel_listener(self):
        """ Using the control message subscription socket, listen to control messages from the `DialogSystem` in a loop.
            Meant to be called in a thread.
        """
        listen = True
        while listen:
            try:
                # receive message for subscribed control topic
                msg = self._control_channel_sub.recv_multipart(copy=True)
                topic = msg[0].decode("ascii")
                timestamp, content = pickle.loads(msg[1])

                if topic == self._start_topic:
                    # initialize dialog state
                    self.dialog_start()
                    # set all listeners of this service to listening mode (block until they are listening)
                    for internal_start_topic in self._internal_start_topics:
                        _send_msg(self._control_channel_pub, internal_start_topic, True)
                        _recv_ack(self._internal_control_channel_sub, internal_start_topic)
                    _send_ack(self._control_channel_pub, self._start_topic)
                elif topic == self._end_topic:
                    # stop all listeners of this service (block until they stopped)
                    for internal_end_topic in self._internal_end_topics:
                        _send_msg(self._control_channel_pub, internal_end_topic, True)
                        _recv_ack(self._internal_control_channel_sub, internal_end_topic, True)
                    self.dialog_end()
                    _send_ack(self._control_channel_pub, self._end_topic)
                elif topic == self._terminate_topic:
                    # terminate all listeners of this service (block until they stopped)
                    for internal_terminate_topic in self._internal_terminate_topics:
                        _send_msg(self._control_channel_pub, internal_terminate_topic, True)
                        _recv_ack(self._internal_control_channel_sub, internal_terminate_topic, True)
                    self.dialog_exit()
                    _send_ack(self._control_channel_pub, self._terminate_topic)
                    listen = False
                elif topic == self._train_topic:
                    self.train()
                    _send_ack(self._control_channel_pub, self._train_topic)
                elif topic == self._eval_topic:
                    self.eval()
                    _send_ack(self._control_channel_pub, self._eval_topic)
                else:
                    if self.debug_logger:
                        self.debug_logger.info("- (Service): received unknown control message from topic", topic,
                                               " with content", content)
            except KeyboardInterrupt:
                break
            except:
                import traceback
                print("ERROR in Service: _control_channel_listener")
                traceback.print_exc()

    def dialog_start(self):
        """ This function is called before the first message to a new dialog is published.
            You should overwrite this function to set/reset dialog-level variables. """
        pass

    def dialog_end(self):
        """ This function is called after a dialog ended (Topics.DIALOG_END message was received).
            You should overwrite this function to record dialog-level information. """
        pass

    def dialog_exit(self):
        """ This function is called when the dialog system is shutting down.
            You should overwrite this function to stop your threads and cleanup any open resources. """
        pass

    def train(self):
        """ Sets module to training mode """
        self.is_training = True

    def eval(self):
        """ Sets module to eval mode """
        self.is_training = False

    def run_standalone(self, host_reg_port: int = 65535):
        """
        Run this service as a standalone serivce (without a `DialogSystem`) on a remote node.
        Use a `RemoteService` with *corresponding identifier* on the `DialogSystem` node to connect both.
        Note: this call is blocking!

        Args:
            host_reg_port (int): The port on the `DialogSystem` node listening for `Service` register requests
        """
        assert self._identifier is not None, "running a service on a remote node requires a unique identifier"
        print("Waiting for dialog system host...")

        # send service info to dialog system node
        self._init_pubsub()
        ctx = Context.instance()
        sync_endpoint = ctx.socket(zmq.REQ)
        sync_endpoint.connect(f"tcp://{self._host_addr}:{host_reg_port}")
        data = pickle.dumps((self._domain_name, self._sub_topics, self._pub_topics, self._start_topic, self._end_topic,
                             self._terminate_topic))
        sync_endpoint.send_multipart((bytes(f"REGISTER_{self._identifier}", encoding="ascii"), data))

        # wait for registration confirmation
        registered = False
        while not registered:
            msg = sync_endpoint.recv()
            msg = msg.decode("utf-8")
            if msg.startswith("ACK_REGISTER_"):
                remote_service_identifier = msg[len("ACK_REGISTER_"):]
                if remote_service_identifier == self._identifier:
                    self._register_with_dialogsystem()
                    sync_endpoint.send_multipart(
                        (bytes(f"CONF_REGISTER_{self._identifier}", encoding="ascii"), pickle.dumps(True)))
                    registered = True
                    print(f"Done")

    def get_all_subscribed_topics(self):
        """
        Returns:
            Set of all topics subscribed to by this `Service`
        """
        return copy.deepcopy(self._sub_topics)

    def get_all_published_topics(self):
        """
        Returns:
            Set of all topics published to by this `Service` 
        """
        return copy.deepcopy(self._pub_topics)

    def _receiver_thread(self, subscriber: Socket, func_instance,
                         topics: Iterable[str], queued_topics: Iterable[str],
                         start_topic: str, end_topic: str, terminate_topic: str):
        """
        Loop for receiving messages.
        Will continue until a message for `terminate_topic` is received.

        Handles waiting for messages, decoding, unpickling and subscription topic to  
        service function keyword mapping.

        Meant to be run in a Thread!

        Args:
            subscriber (Socket): subscriber socket
            func_instance (function instance): the decorated subscriber function instance to be called with the received messages
            topics (Iterable[str]): all last-message-only topics the decorated `func_instance` subscribes to
            queued_topics (Iterable[str]): all collect-all-messages-since-last-call topics the decorated `func_instance` subscribes to
            start_topic (str): Control message topic to set this specific `function_instance` into listening mode (receive all non-control messages)
            end_topic (str): Control message topic to set this specific `function_instance` into non-listening mode (ignore all non-control messages)
            terminate_topic (str): Control message topic to end the listener loop for this specific `function_instance`. 
                                   Also closes the socket before returning.
        """

        ctx = Context.instance()
        control_channel_pub = ctx.socket(zmq.PUB)
        control_channel_pub.sndhwm = 1100000
        control_channel_pub.connect(f"{self._protocol}://{self._host_addr}:{self._pub_port}")

        values = {}
        timestamps = {}
        all_sub_topics = topics + queued_topics
        num_topics = len(all_sub_topics)
        active = False
        terminating = False

        while not terminating:
            try:
                msg = subscriber.recv_multipart(copy=True)
                topic = msg[0].decode("ascii")
                # based on topic, decide what to do
                if topic == start_topic:
                    # reset values and start listening to non-control messages
                    values = {}
                    timestamps = {}
                    active = True
                    _send_ack(control_channel_pub, start_topic)
                elif topic == end_topic:
                    # ignore all non-control messages
                    active = False
                    _send_ack(control_channel_pub, end_topic)
                elif topic == terminate_topic:
                    # shutdown listener thread by exiting loop
                    active = False
                    _send_ack(control_channel_pub, terminate_topic)
                    terminating = True
                else:
                    # non-control message
                    if active:
                        # process message
                        timestamp, content = pickle.loads(msg[1])
                        if self.debug_logger:
                            self.debug_logger.info(
                                f"- (DS): listener thread for function {func_instance}:\n   received for topic {topic}:\n   {content}")

                        # simple synchronization mechanism: remember only newest values,
                        # store them until there was at least 1 new value received per topic.
                        # Then call callback function with complete set of values.
                        # Reset values afterwards and start collecting again.

                        # problem: routing based on prefixes -> function argument names may differ
                        # solution: find longest common prefix of argument name and received topic
                        common_prefix = ""
                        for key in all_sub_topics:
                            if topic.startswith(key) and len(topic) > len(common_prefix):
                                common_prefix = key
                        if common_prefix in topics:
                            # store only latest value
                            values[common_prefix] = content  # set value for received topic
                            timestamps[common_prefix] = timestamp  # set timestamp for received value
                        else:
                            # topic is a queued_topic - queue all values and their timestamps
                            if not common_prefix in values:
                                values[common_prefix] = []
                                timestamps[common_prefix] = []
                            values[common_prefix].append(content)
                            timestamps[common_prefix].append(timestamp)

                        if len(values) == num_topics:
                            # received a new value for each topic -> call callback function
                            if func_instance.timestamp_enabled:
                                # append timestamps, if required
                                values['timestamps'] = timestamps
                            if self.debug_logger:
                                self.debug_logger.info(
                                    f"- (DS): received all messages for function {func_instance}\n   -> CALLING function")
                            if self.__class__ == Service:
                                # NOTE workaround for publisher / subscriber without being an instance method
                                func_instance(**values)
                            else:
                                func_instance(self, **values)
                            # reset values
                            values = {}
                            timestamps = {}
            except KeyboardInterrupt:
                break
            except:
                print("THREAD ERROR")
                import traceback
                traceback.print_exc()
        # shutdown
        subscriber.close()


# Each decorated function should return a dictonary with the keys matching the pub_topics names
def PublishSubscribe(sub_topics: List[str] = [], pub_topics: List[str] = [], queued_sub_topics: List[str] = []):
    """
    Decorator function for services.
    To be able to publish / subscribe to / from topics,
    your class is required to inherit from services.service.Service.
    Then, decorate any function you like.

    Your function will be called as soon as:
        * at least one message is received for each topic in sub_topics (only latest message will be forwarded, others dropped)
        * at least one message is received for each topic in queued_sub_topics (all messages since the previous function call will be forwarded as a list)

    Args:
        sub_topics(List[str or utils.topics.Topic]): The topics you want to get the latest messages from.
                                                     If multiple messages are received until your function is called,
                                                     you will only receive the value of the latest message, previously received
                                                     values will be discarded.
        pub_topics(List[str or utils.topics.Topic]): The topics you want to publish messages to.
        queued_sub_topics(List[str or utils.topics.Topic]): The topics you want to get all messages from.
                                                            If multiple messages are received until your function is called,
                                                            you will receive all values since the previous function call as a list.

    Notes:
        * Subscription topic names have to match your function keywords
        * Your function should return a dictionary with the keys matching your publish topics names 
          and the value being any arbitrary python object or primitive type you want to send
        * sub_topics and queued_sub_topics have to be disjoint!
        * If you need timestamps for your messages, specify a 'timestamps' argument in your subscribing function.
          It will be filled by a dictionary providing timestamps for each received value, indexed by name.
    
    Technical notes:
        * Data will be automatically pickled / unpickled during send / receive to reduce meassage size.
          However, some python objects are not serializable (e.g. database connections) for good reasons
          and will throw an error if you try to publish them.
        * The domain name of your service class will be appended to your publish topics.
          Subscription topics are prefix-matched, so you will receive all messages from 'topic/suffix'
          if you subscibe to 'topic'.
    """

    def wrapper(func):
        def delegate(self, *args, **kwargs):
            func_inst = getattr(self, func.__name__)

            callargs = list(args)
            if self in callargs:    # remove self when in *args, because already known to function
                callargs.remove(self)
            result = func(self, *callargs, **kwargs)
            if result:
                # fix! (user could have multiple "/" characters in topic - only use last one )
                domains = {res.split("/")[0]: res.split("/")[1] if "/" in res else "" for res in result}
                result = {key.split("/")[0]: result[key] for key in result}

            if func_inst not in self._publish_sockets:
                # not a publisher, just normal function
                return result

            socket = self._publish_sockets[func_inst]
            domain = self._domain_name
            if socket and result:
                # publish messages
                for topic in pub_topics:
                # for topic in result: # NOTE publish any returned value in dict with it's key as topic
                    if topic in result:
                        domain = domain if domain else domains[topic]
                        topic_domain_str = f"{topic}/{domain}" if domain else topic
                        if topic in self._pub_topic_domains:
                            topic_domain_str = f"{topic}/{self._pub_topic_domains[topic]}" if self._pub_topic_domains[topic] else topic
                        _send_msg(socket, topic_domain_str, result[topic])
                        if self.debug_logger:
                            self.debug_logger.info(
                                f"- (DS): sent message from {func} to topic {topic_domain_str}:\n   {result[topic]}")
            return result

        # declare function as publish / subscribe functions and attach the respective topics
        delegate.pubsub = True
        delegate.sub_topics = sub_topics
        delegate.queued_sub_topics = queued_sub_topics
        delegate.pub_topics = pub_topics
        # check arguments: is subsriber interested in timestamps?
        delegate.timestamp_enabled = 'timestamps' in inspect.getfullargspec(func)[0]

        return delegate

    return wrapper


class DialogSystem:
    """
    This class will constrct a dialog system from the list of services provided to the constructor.
    It will also handle synchronization for initalization of services before dialog start / after dialog end / on system shutdown
    and lets you discover potential conflicts in you messaging pipeline.
    This class is also used to communicate / synchronize with services running on different nodes.
    """

    def __init__(self, services: List[Union[Service, RemoteService]], sub_port: int = 65533, pub_port: int = 65534,
                 reg_port: int = 65535, protocol: str = 'tcp', debug_logger: DiasysLogger = None):
        """
        Args:
            services (List[Union[Service, RemoteService]]): List of all (remote) services to connect to.
                                                            Only once they're specified here will they start listening for
                                                            messages.
            sub_port(int): subscriber port
            sub_addr(str): IP-address or domain name of proxy subscriber interface (e.g. 127.0.0.1 for your local machine)
            pub_port(int): publisher port
            pub_addr(str): IP-address or domain name of proxy publisher interface (e.g. 127.0.0.1 for your local machine) 
            reg_port (int): registration port for remote services
            protocol(str): communication protol, either 'inproc' or 'tcp' or `ipc`
            debug_logger (DiasysLogger): If not `None`, all messags are printed to the logger, including send/receive events.
                                Can be useful for debugging because you can still see messages received by the `DialogSystem`
                                even if they are never forwarded (as expected) to your `Service`
        """
        # node-local topics
        self.debug_logger = debug_logger
        self.protocol = protocol
        self._sub_topics = {}
        self._pub_topics = {}
        self._remote_identifiers = set()
        self._services = []  # collects names and instances of local services
        self._start_dialog_services = set()  # collects names of local services that subscribe to dialog_start

        # node-local sockets
        self._domains = set()

        # start proxy thread
        self._proxy_dev = ProcessProxy(in_type=zmq.XSUB, out_type=zmq.XPUB)  # , mon_type=zmq.XSUB)
        self._proxy_dev.bind_in(f"{protocol}://127.0.0.1:{pub_port}")
        self._proxy_dev.bind_out(f"{protocol}://127.0.0.1:{sub_port}")
        self._proxy_dev.start()
        self._sub_port = sub_port
        self._pub_port = pub_port

        # thread control
        self._start_topics = set()
        self._end_topics = set()
        self._terminate_topics = set()
        self._stopEvent = threading.Event()

        # control channels
        ctx = Context.instance()
        self._control_channel_pub = ctx.socket(zmq.PUB)
        self._control_channel_pub.sndhwm = 1100000
        self._control_channel_pub.connect(f"{protocol}://127.0.0.1:{pub_port}")
        self._control_channel_sub = ctx.socket(zmq.SUB)

        # register services (local and remote)
        remote_services = {}
        for service in services:
            if isinstance(service, Service):
                # register local service
                service_name = type(service).__name__ if service._identifier is None else service._identifier
                service._init_pubsub()
                self._add_service_info(service_name, service._domain_name, service._sub_topics, service._pub_topics,
                                       service._start_topic, service._end_topic, service._terminate_topic)
                service._register_with_dialogsystem()
            elif isinstance(service, RemoteService):
                remote_services[getattr(service, 'identifier')] = service
        self._register_remote_services(remote_services, reg_port)

        self._control_channel_sub.connect(f"{protocol}://127.0.0.1:{sub_port}")
        self._setup_dialog_end_listener()

        time.sleep(0.25)

    def _register_pub_topic(self, publisher, topic: str):
        """ Map a publisher instance to a topic """
        if not topic in self._pub_topics:
            self._pub_topics[topic] = set()
        self._pub_topics[topic].add(publisher)

    def _register_sub_topic(self, subscriber, topic):
        """ Map a subscriber instance to a topic """
        if not topic in self._sub_topics:
            self._sub_topics[topic] = set()
        self._sub_topics[topic].add(subscriber)

    def _register_remote_services(self, remote_services: List[RemoteService], reg_port: int):
        """ 
        Register all remote services.
        *Blocking* until an ACK was received from all of them, confirming they're setup and ready.

        Args:
            remote_services (List[RemoteService]): list of all remote services to register
            reg_port (int): registration port for remote services
        """
        if len(remote_services) == 0:
            return  # nothing to register

        # Socket to receive registration requests
        ctx = Context.instance()
        reg_service = ctx.socket(zmq.REP)
        reg_service.bind(f'tcp://127.0.0.1:{reg_port}')

        while len(remote_services) > 0:
            # call next remote service
            msg, data = reg_service.recv_multipart()
            msg = msg.decode("utf-8")
            if msg.startswith("REGISTER_"):
                # make sure we have a register message
                remote_service_identifier = msg[len("REGISTER_"):]
                if remote_service_identifier in remote_services:
                    print(f"registering service {remote_service_identifier}...")
                    # add remote service interface info
                    domain_name, sub_topics, pub_topics, start_topic, end_topic, terminate_topic = pickle.loads(data)
                    self._add_service_info(remote_service_identifier, domain_name, sub_topics, pub_topics, start_topic,
                                           end_topic, terminate_topic)
                    self._remote_identifiers.add(remote_service_identifier)
                    # acknowledge service registration
                    reg_service.send(bytes(f'ACK_REGISTER_{remote_service_identifier}', encoding="ascii"))
            elif msg.startswith("CONF_REGISTER_"):
                # complete registration
                remote_service_identifier = msg[len("CONF_REGISTER_"):]
                if remote_service_identifier in remote_services:
                    del remote_services[remote_service_identifier]
                    print(f"successfully registered service {remote_service_identifier}")
                reg_service.send(bytes(f"", encoding="ascii"))
        print("########## Finished registering all remote services ##########")

    def _add_service_info(self, service_name: str, domain_name: str, sub_topics: List[str], pub_topics: List[str], 
                            start_topic: str, end_topic:str, terminate_topic: str):
        """ Add all relevant info from a service (needed to construct dialog graph for debugging).
            Also, sets up all required control channels for this service based on the service's info.
            
        Args:
            service_name (str): service name
            domain_name (str): domain name
            sub_topics (List[str]): list of all subscribed to topics of the given service
            pub_topics (List[str]): list of all topics the given service publishes to
            start_topic (str): control channel topic for setting given service into `listening` mode
            end_topic (str): control channel topic for setting given service into `non-listening` mode
            terminate_topic (str): control channel topic for stopping given service's listener loops and
                                   closing the listener sockets
        """
        self._domains.add(domain_name)
        for topic in sub_topics:
            self._register_sub_topic(service_name, topic)
        for topic in pub_topics:
            self._register_pub_topic(service_name, topic)

        # setup control channels
        self._start_topics.add(start_topic)
        self._end_topics.add(end_topic)
        self._terminate_topics.add(terminate_topic)

        self._control_channel_sub.setsockopt(zmq.SUBSCRIBE, bytes(f"ACK/{start_topic}", encoding="ascii"))
        self._control_channel_sub.setsockopt(zmq.SUBSCRIBE, bytes(f"ACK/{end_topic}", encoding="ascii"))
        self._control_channel_sub.setsockopt(zmq.SUBSCRIBE, bytes(f"ACK/{terminate_topic}", encoding="ascii"))

    def _setup_dialog_end_listener(self):
        """ Creates socket for listening to Topic.DIALOG_END messages """
        ctx = Context.instance()
        self._end_socket = ctx.socket(zmq.SUB)
        # subscribe to dialog end from all domains
        self._end_socket.setsockopt(zmq.SUBSCRIBE, bytes(Topic.DIALOG_END, encoding="ascii"))
        self._end_socket.connect(f"{self.protocol}://127.0.0.1:{self._sub_port}")

        # # add to list of local topics
        # if Topic.DIALOG_END not in self._local_sub_topics:
        #     self._local_sub_topics[Topic.DIALOG_END] = set()
        # self._local_sub_topics[Topic.DIALOG_END].add(type(self).__name__)

    def stop(self):
        """ Set stop event (can be queried by services via the `terminating()` function) """
        self._stopEvent.set()
        pass

    def terminating(self):
        """ Returns True if the system is stopping, else False """
        return self._stopEvent.is_set()

    def shutdown(self):
        """ Shutdown dialog system.
            This will trigger `terminate` messages to be sent to all registered services to stop their listener loops.
            Should be called in the end before exiting your program.
            Blocks until all services sent ACK's confirming they're stopped.
        """
        self._stopEvent.set()
        for terminate_topic in self._terminate_topics:
            _send_msg(self._control_channel_pub, terminate_topic, True)
            _recv_ack(self._control_channel_sub, terminate_topic)

    def _end_dialog(self):
        """ Block until all receivers stopped listening.
            Then, calls `dialog_end` on all registered services. """

        # listen for Topic.DIALOG_END messages
        while True:
            try:
                msg = self._end_socket.recv_multipart(copy=True)
                # receive message for subscribed topic
                topic = msg[0].decode("ascii")
                timestamp, content = pickle.loads(msg[1])
                if content:
                    if self.debug_logger:
                        self.debug_logger.info(f"- (DS): received DIALOG_END message in _end_dialog from topic {topic}")
                    self.stop()
                    break
            except KeyboardInterrupt:
                break
            except:
                import traceback
                traceback.print_exc()
                print("ERROR in _end_dialog ")

        # stop receivers (blocking)
        for end_topic in self._end_topics:
            _send_msg(self._control_channel_pub, end_topic, True)
            _recv_ack(self._control_channel_sub, end_topic)
        if self.debug_logger:
            self.debug_logger.info(f"- (DS): all services STOPPED listening")

    def _start_dialog(self, start_signals: dict):
        """ Block until all receivers started listening.
            Then, call `dialog_start`on all registered services.
            Finally, publish all start signals given. """
    
        self._stopEvent.clear()
        if platform.system().lower() == 'windows':
            time.sleep(1) # wait until stop event is cleared and dialog system is listening
        # start receivers (blocking)
        for start_topic in self._start_topics:
            _send_msg(self._control_channel_pub, start_topic, True)
            _recv_ack(self._control_channel_sub, start_topic)
        if self.debug_logger:
            self.debug_logger.info(f"- (DS): all services STARTED listening")
        # publish first turn trigger
        # for domain in self._domains:
        # "wildcard" mechanism: publish start messages to all known domains
        for topic in start_signals:
            _send_msg(self._control_channel_pub, f"{topic}", start_signals[topic])

    def run_dialog(self, start_signals: dict = {Topic.DIALOG_END: False}):
        """ Run a complete dialog (blocking).
            Dialog will be started via messages to the topics specified in `start_signals`.
            The dialog will end on receiving any `Topic.DIALOG_END` message with value 'True',
            so make sure at least one service in your dialog graph will publish this message eventually.

        Args:
            start_signals (Dict[str, Any]): mapping from topic -> value
                                            Publishes the value given for each topic to the respective topic.
                                            Use this to trigger the start of your dialog system.
        """
        self._start_dialog(start_signals)
        self._end_dialog()

    def list_published_topics(self):
        """ Get all declared publisher topics.

        Returns:
            A dictionary with mapping
                topic (str) -> publishing services (Set[str]).
        Note:
            * Call this method after instantiating all services.
            * Even though a publishing topic might be listed here, there is no guarantee that
            its publisher(s) might ever publish to it.
        """
        return copy.deepcopy(self._pub_topics)  # copy s.t. no user changes this list

    def list_subscribed_topics(self):
        """ Get all declared subscribed topics.

        Returns:
            A dictionary with mapping 
                topic (str) -> subscribing services (Set[str]).
        Notes:
            * Call this method after instantiating all services.
        """
        return copy.deepcopy(self._sub_topics)  # copy s.t. no user changes this list

    def draw_system_graph(self, name: str = 'system', format: str = "png", show: bool = True):
        """ Draws a graph of the system as a directed graph.
            Services are represented by nodes, messages by directed edges (from publisher to subscriber).
            Warnings are drawn as yellow edges (and the missing subscribers represented by an 'UNCONNECTED SERVICES' node),
            errors as red edges (and the missing publishers represented by the 'UNCONNECTED SERVICES' node as well).
            Will mark remote services with blue.

        Args:
            name (str): used to construct the name of your output file
            format (str): output file format (e.g. png, pdf, jpg, ...)
            show (bool): if True, the graph image will be opened in your default image viewer application

        Requires:
            graphviz library (pip install graphviz)
        """
        from graphviz import Digraph
        g = Digraph(name=name, format=format)

        # collect all services, errors and warnings
        services = set()
        for service_set in self._pub_topics.values():
            services = services.union(service_set)
        for service_set in self._sub_topics.values():
            services = services.union(service_set)
        errors, warnings = self.list_inconsistencies()

        # add services as nodes
        for service in services:
            if service in self._remote_identifiers:
                g.node(service, color='#1f618d', style='filled', fontcolor='white', shape='box')  # remote service
            else:
                g.node(service, color='#1c2833', shape='box')  # local service
        if len(errors) > 0 or len(warnings) > 0:
            g.node('UNCONNECTED SERVICES', style='filled', color='#922b21', fontcolor='white', shape='box')

        # draw connections from publisher to subscribers as edges
        for topic in self._pub_topics:
            publishers = self._pub_topics[topic]
            receivers = self._sub_topics[topic] if topic in self._sub_topics else []
            for receiver in receivers:
                for publisher in publishers:
                    g.edge(publisher, receiver, label=topic)

        # draw warnings and errors as edges to node 'UNCONNECTED SERVICES'
        for topic in errors:
            receivers = errors[topic]
            for receiver in receivers:
                g.edge('UNCONNECTED SERVICES', receiver, color='#c34400', fontcolor='#c34400', label=topic)
        for topic in warnings:
            publishers = warnings[topic]
            for publisher in publishers:
                g.edge(publisher, 'UNCONNECTED SERVICES', color='#e37c02', fontcolor='#e37c02', label=topic)

        # draw graph
        g.render(view=show, cleanup=True)

    def list_inconsistencies(self):
        """ Checks for potential errors in the current messaging pipleline:
        e.g. len(list_inconsistencies()[0]) == 0 -> error free pipeline

        (Potential) Errors are defined in this context as subscribed topics without publishers.
        Warnings are defined in this context as published topics without subscribers.

        Returns:
            A touple of dictionaries:
            * the first dictionary contains potential errors (with the mapping topics -> subsribing services) 
            * the second dictionary contains warnings (with the mapping topics -> publishing services).
        Notes:
            * Call this method after instantiating all services.
            * Even if there are no errors returned by this method, there is not guarantee that all publishers 
            eventually publish to their respective topics.
        """
        # look for subscribers w/o publishers by checking topic prefixes
        errors = {}
        for sub_topic in self._sub_topics:
            found_pub = False
            for pub_topic in self._pub_topics:
                if pub_topic.startswith(sub_topic):
                    found_pub = True
                    break
            if not found_pub:
                errors[sub_topic] = self._sub_topics[sub_topic]
        # look for publishers w/o subscribers by checking topic prefixes
        warnings = {}
        for pub_topic in self._pub_topics:
            found_sub = False
            for sub_topic in self._sub_topics:
                if pub_topic.startswith(sub_topic):
                    found_sub = True
                    break
            if not found_sub:
                warnings[pub_topic] = self._pub_topics[pub_topic]

        return errors, warnings

    def print_inconsistencies(self):
        """ Checks for potential errors in the current messaging pipleline:
        e.g. len(list_local_inconsistencies()[0]) == 0 -> error free pipeline and prints them 
        to the console.

        (Potential) Errors are defined in this context as subscribed topics without publishers.
        Warnings are defined in this context as published topics without subscribers.

        Notes:
            * Call this method after instantiating all services.
            * Even if there are no errors returned by this method, there is not guarantee that all publishers 
            eventually publish to their respective topics.
        """
        # console colors
        WARNING = '\033[93m'
        ERROR = '\033[91m'
        ENDC = '\033[0m'

        errors, warnings = self.list_inconsistencies()
        print(ERROR)
        print("(Potential) Errors (subscribed topics without publishers):")
        for topic in errors:
            print(f"  topic: '{topic}', subscribed to in services: {errors[topic]}")
        print(ENDC)
        print(WARNING)
        print("Warnings (published topics without subscribers):")
        for topic in warnings:
            print(f"  topic: '{topic}', published in services: {warnings[topic]}")
        print(ENDC)

    def is_error_free_messaging_pipeline(self) -> bool:
        """ Checks the current messaging pipeline for potential errors.

        (Potential) Errors are defined in this context as subscribed topics without publishers.

        Returns:
            True, if no potential errors could be found - else, False

        Notes:
            * Call this method after instantiating all services.
            * Lists only node-local (or process-local) inconsistencies.
            * Even if there are no errors returned by this method, there is not guarantee that all publishers 
            eventually publish to their respective topics.
        """
        return len(self.list_inconsistencies()[0]) == 0
