import asyncio
import functools
from multiprocessing import Queue, Process, Manager
from message_codes import MessageCode
from uri import URI
import websockets
import random
from typing import Dict, Iterable, List, Tuple, Union
import json
import traceback
import logging
import logging.handlers
import sys

"""
Supported serialization formats:
* JSON: yes -> wamp.2.json, UTF8 encoded payload
* (MessagePack: no) -> wamp.2.msgpack, binary encoded 

Supported transports:
* WebSocket: yes
"""


"""
* Sessions (global scope), Publications (global scope), Subscriptions (router scope), Registrations (router scope), Requests (session scope) are identified in WAMP using IDs that are integers between (inclusive) 1 and 2^53 
* IDs in the global scope MUST be drawn randomly from a uniform distribution over the complete range [1, 2^53]
* IDs in the router scope CAN be chosen freely by the specific router implementation
* IDs in the session scope MUST be incremented by 1 beginning with 1 (for each direction - Client-to-Router and Router-to-Client)
"""





"""
All WAMP messages are a list with a first element MessageType followed by one or more message type specific elements:
  [MessageType|integer, ... one or more message type specific elements ...]

Example:
	SUBSCRIBE message format:
	[SUBSCRIBE, Request|id, Options|dict, Topic|uri]

	Example message:
	[32, 713845233, {}, "com.myapp.mytopic1"]

The application payload (that is call arguments, call results, event payload etc) is always at the end of the message element list
-> Brokers and Dealers have no need to inspect (parse) the application payload
"""





def skip_whitespace(string: str, start_idx: int) -> int:
	""" Return index of next token after potential whitespace """
	while string[start_idx] == " ":
		start_idx += 1
	return start_idx

def read_json_start_token(string: str, start_idx: int) -> int:
	""" Return index of first token after json start token { """
	start_idx = skip_whitespace(string, start_idx)
	if string[start_idx] == '{':
		return start_idx + 1
	raise Exception(f"Invalid start token for json string '{string}' at position {start_idx}")

def read_list_start_token(string: str, start_idx: int) -> int:
	""" Return index of first token after list token [ """
	start_idx = skip_whitespace(string, start_idx)
	if string[start_idx] == '[':
		return start_idx + 1
	raise Exception(f"Invalid start token for json list in string '{string}' at position {start_idx}")

def parse_number(serialized_msg: str, start_idx: int) -> Tuple[int, int]:
	""" Returns parsed number and index of first token after number """
	start_idx = skip_whitespace(serialized_msg, start_idx)
	number = []
	while serialized_msg[start_idx] != ',':
		number.append(serialized_msg[start_idx])
		start_idx += 1
	number = int("".join(number).strip())
	return number, start_idx

def parse_string(serialized_msg: str, start_idx: int) -> Tuple[int, int]:
	""" Returns parsed number and index of first token after number """
	while serialized_msg[start_idx] != '"':
		start_idx += 1
	start_idx += 1 # skip opening "
	begin = start_idx
	while serialized_msg[start_idx] != '"':
		start_idx += 1
	return serialized_msg[begin:start_idx], start_idx + 1 # skipt closing "

def parse_msg_type(serialized_msg: str) -> Tuple[MessageCode, int]:
	""" Returns parsed message type and index of first token after type id"""
	start_idx = read_list_start_token(serialized_msg, 0)
	msg_type_id, start_idx = parse_number(serialized_msg, start_idx)
	return MessageCode(msg_type_id), start_idx + 1 # skip comma

def skip_argument(seralized_msg: str, start_idx: int) -> int:
	""" Returns start index of next argument (arguments are seperated by ,)"""
	while seralized_msg[start_idx] != ',':
		start_idx += 1
	if seralized_msg[start_idx] == ',':
		start_idx += 1
	return start_idx

def draw_free_random_id(id_map: Iterable):
	id = random.randint(1, 2**53)
	while id in id_map:
		id = random.randint(1, 2**53)	
	return id

	
class Router:
	def __init__(self, url: str = 'localhost', port: int = 44122):
		self.sessions = {}
		self.topic_to_subscription_id = {}
		self.topic_subscribers = {}
		self.url = url
		self.port = port

	def serve(self, clients_to_connect: list, logger_queue: Queue, log_level: int):
		h = logging.handlers.QueueHandler(logger_queue)
		root = logging.getLogger()
		root.addHandler(h)
		root.setLevel(log_level)
		self.logger = logging.getLogger("router")

		# loop = asyncio.new_event_loop()
		bound_handler = functools.partial(self.msg_loop, clients_to_connect=clients_to_connect)
		start_server = websockets.serve(bound_handler, self.url, self.port, ping_interval=None, subprotocols=['wamp.2.json'])
		# loop.run_until_complete(start_server)
		# loop.run_forever()
		# asyncio.create_task(start_server)
		# asyncio.get_running_loop().create_task(start_server)
		self.logger.info('router starting msg loop')
		# await start_server.wait_closed()
		# asyncio.get_running_loop().run_until_complete(start_server)
		# asyncio.get_running_loop().run_forever()
		loop = asyncio.get_event_loop()
		self.logger.info(f'loop id {id(loop)}')
		loop.run_until_complete(start_server)
		loop.run_forever()

	async def _reply_welcome(self, socket):
		# NOTE: we discard realm info here
		# reply with random session id
		session_id = draw_free_random_id(self.sessions)
		reply = [MessageCode.WELCOME.value, session_id, {"roles": {"broker": {}}}]
		self.sessions[session_id] = socket # save connection
		await socket.send(json.dumps(reply))

	async def _confirm_subscription(self, socket, serialized_msg):
		# Client msg: [MessageCode.SUBSCRIBE.value, request_id, {}, topic]
		msg = json.loads(serialized_msg)
		_, request_id, _, topic = msg

		# confirm subscription
		try:
			topic = URI(topic).uri
			self.logger.info(f'TRY SUB {topic}')
			if not topic in self.topic_to_subscription_id:
				# new topic - assign id
				subscription_id = draw_free_random_id(self.topic_to_subscription_id.values())
				self.topic_to_subscription_id[topic] = subscription_id
				self.topic_subscribers[topic] = []
			else:
				subscription_id = self.topic_to_subscription_id[topic]
			if not socket in self.topic_subscribers[topic]:
				self.topic_subscribers[topic].append(socket)
			reply = [MessageCode.SUBSCRIBED.value, request_id, subscription_id]
			self.logger.info(f'added subscriber, current subscriber list: {self.topic_subscribers}')
			await socket.send(json.dumps(reply))
		except:
			# reply error message, e.g. could be invalid URI or other problem
			self.logger.error(f"Error while subscribing to URI {topic}")
			traceback.print_exc()
			await socket.send(json.dumps(MessageCode.ERROR.value, MessageCode.SUBSCRIBE.value, request_id, {}, "wamp.error.invalid_uri"))

	async def _publish_message(self, serialized_msg, msg_type_end_idx):
		# published message: [MessageCode.PUBLISH, request_counter, {}, topic, args, kwargs]
		# parse string until reaching topic. we already parsed message code
		start_idx = skip_argument(serialized_msg, msg_type_end_idx) # skip request counter
		start_idx = skip_argument(serialized_msg, start_idx) # skip options
		try:
			topic, start_idx = parse_string(serialized_msg, start_idx)
			topic = URI(topic).uri # parse topic
		except:
			raise Exception(f"Published topic {topic} is not a valid uri")
		self.logger.info(f'publishing {topic}')
		
		# forward es event to each subscriber
		publication_id = random.randint(1, 2**53)
		if topic in self.topic_subscribers:
			for sub_socket in self.topic_subscribers[topic]:
				self.logger.info(f"publishing to {topic}, {sub_socket}")
				msg = f"[{MessageCode.EVENT.value},{self.topic_to_subscription_id[topic]},{publication_id}" + ",{}" + serialized_msg[start_idx:]
				asyncio.create_task(sub_socket.send(msg))


	async def msg_loop(self, websocket: websockets.WebSocketServerProtocol, path: str, clients_to_connect: list):
		# handle new client: should start with hello message
		try:
			serialized_msg = await websocket.recv()
			print("ROUTER RECV", serialized_msg)
			msg_type, msg_type_end_idx = parse_msg_type(serialized_msg)
			if msg_type != MessageCode.HELLO:
				return websocket.close() # protocol violation
			# read details -> identifier of connecting client
			self.logger.info(f'connecting client, waiting for {clients_to_connect}')
			if 'identifier' in json.loads(serialized_msg)[2]:
				client_id = json.loads(serialized_msg)[2]['identifier']
				if client_id in clients_to_connect:
					self.logger.info(f'connected client {client_id}')
					clients_to_connect.remove(client_id)
			else:
				# TESTCODE ONLY: FOR AUTOBAHN CONNECTION
				client_id = 'webapp'
				if client_id in clients_to_connect:
					self.logger.info(f'connected client {client_id}')
					clients_to_connect.remove(client_id)
			await self._reply_welcome(websocket) # reply with welcome msg
			# self.logger.info(f'connected to client {websocket}')

			# message loop
			async for serialized_msg in websocket:
				# check message type and process accordingly
				# format is a json-list, first entry is message type
				# don't parse full json here for speed, just route message
				self.logger.info(f'recv message {serialized_msg}')
				msg_type, msg_type_end_idx = parse_msg_type(serialized_msg)
				if msg_type == MessageCode.HELLO:
					websocket.close() # protocol violation
				elif msg_type == MessageCode.SUBSCRIBE:
					await self._confirm_subscription(websocket, serialized_msg)
				elif msg_type == MessageCode.PUBLISH:
					await self._publish_message(serialized_msg, msg_type_end_idx)
				if msg_type == MessageCode.GOODBYE:
					self.logger.info('closing')
					websocket.close()
		except:
			self.logger.error("ERROR in message loop:")
			traceback.print_exc()
			self.logger.info("Closing connection...")
			await websocket.close()

# if __name__ == '__main__':
# 	router = Router()
# 	router.run_blocking()