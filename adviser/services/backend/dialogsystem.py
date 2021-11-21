import asyncio
from service import RemoteService, Service
from client import Client
import json
from message_codes import MessageCode
import multiprocessing
from multiprocessing import Manager
from typing import Dict, Union, List
import time
import logging
import logging.handlers
import sys
from uri import URI

from router import Router, parse_msg_type

class DiasysLogger:
    pass
class Domain:
    pass



def logger_configurer():
    root = logging.getLogger()
    h = logging.StreamHandler()
    h.flush = sys.stdout.flush
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)


def logger_process(queue):
    logger_configurer()
    while True:
        try:
            record = queue.get()
            if record is None: 
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)



class Dialogsystem:
    def __init__(self, services=[], url='localhost', port=44122, log_level: int = logging.INFO) -> None:
        self.services = services

        # logging
        self.logger_queue = multiprocessing.Queue(-1)
        self.logging_proc = multiprocessing.Process(target=logger_process, args=(self.logger_queue,))
        self.logging_proc.start()
        h = logging.handlers.QueueHandler(self.logger_queue)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(log_level)
        self.logger = logging.getLogger("dialogsystem")

        # with Manager() as manager:
        self.manager = Manager()
        clients_to_connect = self.manager.list()
        for service in services:
            if isinstance(service, Service):
                service._identifier = str(id(service)) # assign new ID
                clients_to_connect.append(service._identifier)
            elif isinstance(service, RemoteService):
                clients_to_connect.append(service._identifier) # use remote id


        # start router
        self.router = Router(url=url, port=port)
        proc = multiprocessing.Process(target=self.router.serve, args=(clients_to_connect, self.logger_queue, log_level))
        proc.start()

        # self.ds_client = Client()
        self.logger.info('Connecting services...')

        for service in self.services:
            if isinstance(service, Service):
                service.run(self.logger_queue, log_level)
                self.logger.info(f'started service {service}')
            # elif isinstance(service, RemoteService):
                # pass # wait for remote handshake
        while len(clients_to_connect) > 0:
            time.sleep(0.1) # wait for services to register
        self.logger.info("ALL CLIENTS CONNECTED")
        self.ds_client = Client(identifier='dialogsystem', url=url, port=port)
        # time.sleep(3)

    async def _start(self, start_msgs: Dict[str, dict]):
        self.ds_client.logger = self.logger
        await self.ds_client._connect(False)
        self.logger.info('ds client connected')

        await self.ds_client.publish(URI('dialog_start').uri, {})
        
        start_tasks = []
        for start_topic in start_msgs:
            # msg = json.dumps([MessageCode.PUBLISH.value, 1, {}, URI(start_topic).uri, [], start_msgs[start_topic]])
            # _, msg_type_end_idx = parse_msg_type(msg)
            start_tasks.append(asyncio.create_task(self.ds_client.publish(URI(start_topic).uri, start_msgs[start_topic])))
        await asyncio.wait(start_tasks)
        self.logger.info('all start messages triggered')

        async for msg in self.ds_client.websocket:
            # removing this code will close the ds_client socket - find a way to deal with that
            print('server msg', msg)

    def start(self, start_msgs: Dict[str, dict]):
        asyncio.run(self._start(start_msgs))

