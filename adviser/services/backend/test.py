

import asyncio
from time import asctime
from dialogsystem import Dialogsystem
from service import Service, PublishSubscribe, RemoteService

class TestService(Service):
    # def __init__(self, domain = "", sub_topic_domains = {}, pub_topic_domains = {},
	# 			ds_host_addr: str = "localhost", ds_host_port: int = 44122, debug_logger = None, identifier: str = None) -> None:
    #     super().__init__(domain, sub_topic_domains, pub_topic_domains, ds_host_addr, ds_host_port, debug_logger, identifier)

    @PublishSubscribe(sub_topics=['start'])
    def callback(self, start=None):
        print("CALLBACK from REMOTE SERVICE", start)

		
if __name__ == "__main__":
    testService = TestService(identifier='webapp')
    testService.run()

	