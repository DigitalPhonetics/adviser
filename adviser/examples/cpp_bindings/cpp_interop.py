import sys
import os
sys.path.append(os.path.realpath("../.."))

from services.service import Service, PublishSubscribe, DialogSystem, RemoteService
from utils.topics import Topic
from zmq import Context
import zmq
from typing import List, Union, Dict
from threading import Thread
import time

class CPPRemoteService(Service):
    def __init__(self, domain = "", sub_topic_domains: Dict[str, str] = {}, pub_topic_domains: Dict[str, str] = {},
                 ds_host_addr: str = "127.0.0.1", sub_port: int = 65533, pub_port: int = 65534, protocol: str = "tcp",
                 debug_logger = None, identifier: str = None,
                 cpp_service_addr: str = "127.0.0.1", cpp_service_sync_port: int = 6006, cpp_service_sub_port: int = 6007, cpp_service_pub_port: int = 6008):
        """
        NOTE:
            cpp_service_addr: add the IP address of your remote machine running the C++ service here
            cpp_service_port: add the port you specified in the C++ service (and opened on the remote machine!) here
        """
        super().__init__(domain, sub_topic_domains, pub_topic_domains, ds_host_addr, sub_port, pub_port, protocol, debug_logger, identifier)
        
        self._cpp_service_adr = cpp_service_addr
        self._cpp_service_sync_port = cpp_service_sync_port
        self._cpp_service_sub_port = cpp_service_sub_port
        self._cpp_service_pub_port = cpp_service_pub_port

        self._CPP_TERMINATE_TOPIC = "CPPService/TERMINATE"
        self.listen = False


    def _register_with_dialogsystem(self):
        super()._register_with_dialogsystem()

        # establish connection to C++ service here
        cpp_sync_url = f"tcp://{self._cpp_service_adr}:{self._cpp_service_sync_port}"
        cpp_sub_url = f"tcp://{self._cpp_service_adr}:{self._cpp_service_pub_port}"
        cpp_pub_url = f"tcp://{self._cpp_service_adr}:{self._cpp_service_sub_port}"
        ctx = Context.instance()

        self.cpp_sub_endpoint = ctx.socket(zmq.SUB)
        self.cpp_sub_endpoint.setsockopt(zmq.SUBSCRIBE, bytes(self._CPP_TERMINATE_TOPIC, encoding="ascii"))
        self.cpp_sub_endpoint.setsockopt(zmq.SUBSCRIBE, bytes("CONTENT", encoding="ascii"))
        self.cpp_sub_endpoint.connect(cpp_sub_url)

        self.cpp_pub_endpoint = ctx.socket(zmq.PUB)
        self.cpp_pub_endpoint.connect(cpp_pub_url)

        self.cpp_sync_endpoint = ctx.socket(zmq.REQ)
        self.cpp_sync_endpoint.connect(cpp_sync_url)

        ## synchronize (block until ack received)
        print("Connecting to C++ service at", cpp_sync_url)
        self.cpp_sync_endpoint.send(bytes("SYNC", encoding="ascii"))
        ready = False
        while not ready:
            msg = self.cpp_sync_endpoint.recv()
            if msg.decode("ascii") == "ACK_SYNC":
                ready = True
        self.listen = True
        print("Done")

        Thread(target=self.cpp_service_listener).start()
        time.sleep(1.0)

    def dialog_exit(self):
        super().dialog_exit()
        self.listen = False
        print("Dialog exit: waiting for C++ service...")
        self.cpp_sub_endpoint.send_multipart((bytes(self._CPP_TERMINATE_TOPIC_TOPIC, encoding="ascii"), True))

    def cpp_service_listener(self):
        print("Started listener for C++ content messages...")
        while(self.listen):
            msg = self.cpp_sub_endpoint.recv_multipart(copy=True)
            recv_topic = msg[0].decode("ascii")

            if recv_topic == "CONTENT":
                data = msg[1]
                self.forward_from_cpp_service(data)
        print("Stopped listening")

    def send_msg_to_cpp_service(self, topic: str, content: bytes):
        # self.cpp_pub_endpoint.send_multipart((bytes(topic, encoding="ascii"), content))
        self.cpp_pub_endpoint.send_multipart((bytes(topic, encoding="ascii"), content))

    @PublishSubscribe(sub_topics=["next_turn"])
    def forward_to_cpp_service(self, next_turn):
        self.send_msg_to_cpp_service("CONTENT", bytes(next_turn, encoding="ascii"))

    @PublishSubscribe(pub_topics=["topic2"])
    def forward_from_cpp_service(self, content):
        return {"topic2": content}

class OutputService(Service):
    @PublishSubscribe(sub_topics=["topic2"], pub_topics=["next_turn"])
    def print_output(self, topic2):
        print("OUTPUT:", topic2)
        time.sleep(1.0)
        return {"next_turn": "true"}

    
cpp = CPPRemoteService()
out = OutputService()
ds = DialogSystem(services=[cpp, out])

print("DONE SETUP")
ds.run_dialog(start_signals={'next_turn': "true"})
 
ds.shutdown()