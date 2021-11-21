

# Messages

# 1. Hello [HELLO, Realm|uri, Details|dict]
import concurrent
from os import confstr
import traceback
import json
from json import JSONEncoder
import asyncio
import websockets
import random
from typing import Dict, Callable, List, Tuple
import logging

from services.backend.message_codes import MessageCode
from services.backend.uri import URI


class _CustomEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, set):
			return list(obj)
		return super().default(obj)


class Client:	 
	def __init__(self, identifier: str, url: str = 'localhost', port: int = 44122, realm: URI = URI('adviser'), roles = {"publisher": {}, "subscriber": {}}, subscriptions: List[Tuple[str, Callable]]=[]) -> None:
		self._url = url
		self._port = port
		self._roles = roles
		self._realm = realm
		self._subscriptions = {}
		self._request_counter = 1
		self.sub_callbacks = subscriptions
		self._identifier = identifier

	async def subscribe(self, topic: URI, callback_fn):
		self.logger.info(f"subscribing to {topic.uri}")

		# send subscribe request
		request_id = random.randint(1, 2**53)
		msg = [MessageCode.SUBSCRIBE.value, request_id, {}, topic.uri]
		await self.websocket.send(json.dumps(msg, cls=_CustomEncoder))

		# receive subscription answer
		answer_serialized = await self.websocket.recv()
		answer = json.loads(answer_serialized)
		msg_type = MessageCode(answer[0])
		if msg_type == MessageCode.SUBSCRIBED:
			# successful: reply [MessageCode.SUBSCRIBED.value, request_id, subscription_id]
			recv_request_id, subscription_id = answer[1:]
			if recv_request_id != request_id:
				raise Exception("request ids for subscribe request do not match: ", request_id, recv_request_id)
			
			if not subscription_id in self._subscriptions:
				self._subscriptions[subscription_id] = []
			self._subscriptions[subscription_id].append(callback_fn)
			self.logger.info(f'subscription to topic {topic.uri} successful')
		else:
			# not successful
			raise Exception(f'failed to subscribe to topic {topic.uri}')

	# TODO unsubscribe

	async def publish(self, topic: str, kwargs: dict):
		self.logger.info(f'publishing to {topic}: {kwargs}')
		msg = [MessageCode.PUBLISH.value, self._request_counter, {}, topic, [], kwargs]
		self._request_counter += 1
		await self.websocket.send(json.dumps(msg, cls=_CustomEncoder))

	async def _connect(self, register_subscribers: bool):
		connected = False
		while not connected:
			try:
				self.logger.info(f'trying to connect {self._identifier}')
				self.websocket = await websockets.connect(f'ws://{self._url}:{self._port}', ping_interval=None)
				# handshake
				await self._send_hello(self.websocket)
				greeting = await self.websocket.recv()
				connected = True
				self.logger.info('client connected')

				# subscribe
				if register_subscribers:
					for topic, callback_fn in self.sub_callbacks:
						await self.subscribe(URI(topic), callback_fn) # subscribe to all topics
						self.logger.info(f'subscibe to topic {topic} using {callback_fn}')
					self.logger.info("subscribed to all topics")
			except:
				# traceback.print_exc()
				self.logger.info(f'connection for client {self._identifier} failed, trying to reconnect...')
				await asyncio.sleep(0.1)

	async def _msg_loop(self):
		self.logger.info('entering message loop')
		await self._connect(True)
		async for serialized_msg in self.websocket:
			self.logger.info(f'recv: {serialized_msg}')
			msg = json.loads(serialized_msg)
			msg_type = MessageCode(msg[0])
			if msg_type == MessageCode.EVENT:
				sub_id, pub_id, details, args, kwargs = msg[1:]
				for callback_fn in self._subscriptions[sub_id]:
					self.logger.info(f"KWARGS {kwargs}")
					asyncio.create_task(callback_fn(**kwargs))
	
	def start_msg_loop(self, logger_queue, log_level):
		h = logging.handlers.QueueHandler(logger_queue)
		root = logging.getLogger()
		root.addHandler(h)
		root.setLevel(log_level)
		self.logger = logging.getLogger("router")

		loop = asyncio.get_event_loop()
		self.logger.info(f'client loop id {id(loop)}')
		loop.run_until_complete(self._msg_loop())
		loop.run_forever()

	
	async def _send_hello(self, socket):
		# self.logger.info('sending hello...')
		msg = [MessageCode.HELLO.value, self._realm.uri, {"roles": self._roles, "identifier": self._identifier}]
		await socket.send(json.dumps(msg, cls=_CustomEncoder))
