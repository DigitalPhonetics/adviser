

import asyncio
from time import asctime
from dialogsystem import Dialogsystem
from service import Service, PublishSubscribe, RemoteService

class TestService(Service):
	@PublishSubscribe(sub_topics=['start'], pub_topics=['continued'])
	def callback(self, start=None):
		print("CALLBACK from TestService", start)
		# print("TURN", self.turn)
		return {'continued': {"continued": 'CONTINUE MSG'}}

	@PublishSubscribe(sub_topics=['start'])
	async def async_callback(self, start=None):
		print("ASYNC CALLBACK from TestService", start)



class TestService2(Service):
	@PublishSubscribe(sub_topics=['continued'], pub_topics=['c2'])
	def callback(self, continued=None):
		print("CALLBACK from TestService2", continued)
		return {'c2': {"c2": 'C2 MSG'}}


class TestService3(Service):
	@PublishSubscribe(sub_topics=['c2'], pub_topics=['start'])
	async def callback(self, c2=None):
		await asyncio.sleep(3)
		print("CALLBACK from TestService3", c2)
		return {'start': {"start": 'start MSG'}}


		
if __name__ == "__main__":
	testService = TestService()
	testService2 = TestService2()
	testService3 = TestService3()
	# testService4 = RemoteService(identifier='webapp')
	d = Dialogsystem(services=[testService, testService2, testService3]) #, testService4])
	d.start(start_msgs={"start": {}})

	