import logging
import logging.handlers
from typing import Dict, Union, List
from uri import URI
from client import Client
import multiprocessing
import asyncio
import inspect
import functools

class DiasysLogger:
	pass
class Domain:
	pass




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
		* Data will be automatically json-serialized, so make sure it is json-serializable!
		* The domain name of your service class will be appended to your publish topics.
		  Subscription topics are prefix-matched, so you will receive all messages from 'topic.suffix'
		  if you subscibe to 'topic'.
	"""
	def wrapper(func):
		@functools.wraps(func)
		async def delegate(self: Service, *args, **kwargs):
			func_inst = getattr(self, func.__name__)

			callargs = list(args)
			if self in callargs:    # remove self when in *args, because already known to function
				callargs.remove(self)
			if inspect.iscoroutinefunction(func):
				# support for async callbacks
				result = await func(self, *callargs, **kwargs)
			else:
				# support for function callbacks
				result = func(self, *callargs, **kwargs)
			if result:
				domains = {res.split(".")[0]: ".".join(res.split(".")[1:]) if "." in res else "" for res in result}
				result = {key.split(".")[0]: result[key] for key in result}

			if func_inst.__name__ not in self._publish_fn:
				# not a publisher, just normal function
				return result

			# TODO change publish_socket to socket, since there is only 1 socket per service now
			if result:
				domain = self._domain_name
				# publish messages
				for topic in pub_topics:
					# NOTE publish any returned value in dict with it's key as topic
					if topic in result:
						domain = domain if domain else domains[topic]
						topic_domain_str = f"{topic}.{domain}" if domain else topic
						if topic in self._pub_topic_domains:
							topic_domain_str = f"{topic}.{self._pub_topic_domains[topic]}" if self._pub_topic_domains[topic] else topic
						asyncio.create_task(self._client.publish(topic_domain_str, result[topic]))
						if self.debug_logger:
							self.debug_logger.info(f"- (DS): sent message from {func} to topic {topic_domain_str}:\n   {result[topic]}")
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


class Service:
	def __init__(self, domain: Union[str, Domain] = "", sub_topic_domains: Dict[str, str] = {}, pub_topic_domains: Dict[str, str] = {},
				ds_host_addr: str = "localhost", ds_host_port: int = 44122, debug_logger: DiasysLogger = None, identifier: str = None) -> None:
		# connection info
		# domain info
		self.domain = domain
		self._domain_name = self.domain
		self._sub_topic_domains = sub_topic_domains
		self._pub_topic_domains = pub_topic_domains

		self.debug_logger = debug_logger

		self._sub_topics = set()
		self._pub_topics = set()
		self._publish_fn = set()

		self._start_topic = f"{type(self).__name__}.{id(self)}.START"
		self._end_topic = f"{type(self).__name__}.{id(self)}.END"
		self._terminate_topic = f"{type(self).__name__}.{id(self)}.TERMINATE"
		self._train_topic = f"{type(self).__name__}.{id(self)}.TRAIN"
		self._eval_topic = f"{type(self).__name__}.{id(self)}.EVAL"

		self._identifier = str(id(self)) if not identifier else identifier
		subscriber_callbacks = self._init()
		self._client = Client(identifier=self._identifier, url=ds_host_addr, port=ds_host_port, realm=URI('adviser'), roles = {"publisher": {}, "subscriber": {}}, subscriptions=subscriber_callbacks)

	def _init(self): 
		""" Search for all functions decorated with the `PublishSubscribe` decorator and call the setup methods for them.
			Also, set up dialog start, dialog end and dialog exit handlers. """
		sub_callbacks = [
			(URI('dialog_start').uri, self.dialog_start),
			(URI('dialog_end').uri, self.dialog_end),
			(URI('dialog_exit').uri, self.dialog_exit)
		]

		for func_name in dir(self):
			func_inst = getattr(self, func_name)
			if hasattr(func_inst, "pubsub"):
				# found decorated publisher / subscriber function -> setup sockets and listeners
				sub_topics: List[str] = getattr(func_inst, "sub_topics") + getattr(func_inst, 'queued_sub_topics')
				sub_callbacks += [(URI(sub_topic).uri, func_inst) for sub_topic in sub_topics]

				# setup publishers
				if getattr(func_inst, "pub_topics"):
					for pub_topic in getattr(func_inst, "pub_topics"):
						self._pub_topics.add(pub_topic)
						self._publish_fn.add(func_inst.__name__)

		return sub_callbacks
	
	async def dialog_start(self):
		self.logger.info("dialog start")
		pass

	async def dialog_end(self):
		self.logger.info("dialog end")
		pass

	async def dialog_exit(self):
		self.logger.info("dialog exit")
		pass

	def run(self, logger_queue, log_level):
		# Close the connection when receiving SIGTERM.
		#loop = asyncio.get_event_loop()
		#loop.add_signal_handler(signal.SIGTERM, loop.create_task, self._client.websocket.close())

		h = logging.handlers.QueueHandler(logger_queue)
		root = logging.getLogger()
		root.addHandler(h)
		root.setLevel(log_level)
		self.logger = logging.getLogger(f"service {self._client._identifier}")

		proc = multiprocessing.Process(target=self._client.start_msg_loop, args=(logger_queue, log_level))
		proc.start()
		
class RemoteService:
	def __init__(self, identifier: str) -> None:
		self._identifier = identifier
		self._connected = False