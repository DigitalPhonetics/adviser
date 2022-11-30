import asyncio
from collections import defaultdict
from functools import partial
import functools
from inspect import isawaitable
from typing import Any, Callable, Dict, Iterable, Union
from autobahn.asyncio.component import Component, run
from autobahn.wamp import SubscribeOptions
from datetime import datetime
from utils.serializable import JSONSerializable
from utils.domain.domain import Domain
from utils.memory import UserState
import warnings

# TODO where / how to add WAMP subscribe options (wildcard, prefix?) -> prefix should be default I think
# PubSub(topics={"domain1.topicA": "arg1", domain2.topicA": "arg2"}) ==> topic->arg mapping is more flexible than arg->topic


class ControlChannelMessages:
    DIALOG_START = "dialogsystem.start"           # Triggered whenever a new dialog starts. This message has a single argument: user_id
    DIALOG_END = "dialogsystem.end"               # Triggered whenever a dialog was ended. This message has a single argument: user_id

    DIALOGSYSTEM_SHUTDOWN = "dialogsystem.shutdown"     # Triggered whenever a the dialog system should shut down. Will try to stop all services. This message has no arguments.
    DIALOGSYSTEM_STARTUP = "dialogsystem.startup" # TODO add event handlers, fire once DS is set up -> entry loop

    _DIALOG_START_ACK = 'ack.dialogsystem.start'
    _DIALOG_END_ACK = 'ack.dialogsystem.end'



def _serialize(key: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {key: _serialize(inner_key, value[inner_key]) for inner_key in value}
    elif isinstance(value, JSONSerializable):
        return {key: value.to_json()}
    else:
        return {key: value}


class _ServiceFunctionDelegate:
    def __init__(self, fn: Callable, sub_topics: Union[Dict[str, str], Iterable], sub_topics_queued: Union[Dict[str, str], Iterable], pub_topics: Iterable[str], user_id: bool, timestamps: bool) -> None:
        self.fn = fn
        self.fn_name = fn.__name__
        self.sub_topics = sub_topics if isinstance(sub_topics, dict) else {arg: arg for arg in sub_topics}
        self.sub_topics_queued = sub_topics_queued if isinstance(sub_topics_queued, dict) else {arg: arg for arg in sub_topics_queued}
        assert set(self.sub_topics.values()).isdisjoint(self.sub_topics_queued.values()), f"{self.fn_name} ({fn}): The same function argument(s) {set(self.sub_topics.values()).intersection(self.sub_topics_queued.values())} are being used by the both queued and non-queued topics at the same time"
        self.arglist = set(self.sub_topics.values()).union(self.sub_topics_queued.values())
        self.pub_topics = pub_topics
        self.user_id = user_id
        self.timestamps = timestamps
        self.domain_suffix = ""
        # associate memory
        self.call_args_cache = UserState(lambda: defaultdict(lambda: list())) # user -> data: arg_name -> List[values]
        self.timestamp_cache = UserState(lambda: defaultdict(lambda: list())) # user -> data: arg_name -> List[datetime]

    def set_domain_suffix(self, domain_suffix: str):
        self.domain_suffix = domain_suffix
        self.suffixed_sub_topics = {topic + domain_suffix: self.sub_topics[topic] for topic in self.sub_topics}
        self.suffixed_sub_topics_queued = {topic + domain_suffix: self.sub_topics_queued[topic] for topic in self.sub_topics_queued}

    def ready_for_call(self, user_id: int) -> bool:
        """
        Returns True, if at least 1 value was received for each function argument since the last call, else False.
        """
        for arg in self.arglist:
            if len(self.call_args_cache[user_id][arg]) == 0:
                return False
        return True

    def append_values(self, user_id, **kwargs):
        now = datetime.now()
        for topic in kwargs:
            # cache call args
            if topic in self.sub_topics:
                # override values that might already have been published since last function call
                self.call_args_cache[user_id][self.sub_topics[topic]] = [kwargs[topic]]
                if self.timestamps:
                    self.timestamp_cache[user_id][self.sub_topics[topic]] = [now]
            elif topic in self.suffixed_sub_topics:
                # override values that might already have been published since last function call
                self.call_args_cache[user_id][self.suffixed_sub_topics[topic]] = [kwargs[topic]]
                if self.timestamps:
                    self.timestamp_cache[user_id][self.suffixed_sub_topics[topic]] = [now]
            if topic in self.sub_topics_queued:
                # accumulate values between function calls for queued topics
                self.call_args_cache[user_id][self.sub_topics_queued[topic]].append(kwargs[topic])
                if self.timestamps:
                    self.timestamp_cache[user_id][self.sub_topics_queued[topic]].append(now)
            elif topic in self.suffixed_sub_topics_queued:
                # accumulate values between function calls for queued topics
                self.call_args_cache[user_id][self.suffixed_sub_topics_queued[topic]].append(kwargs[topic])
                if self.timestamps:
                    self.timestamp_cache[user_id][self.suffixed_sub_topics_queued[topic]].append(now)


    async def call(self, other, user_id: int = 0, **kwargs) -> Any:
        # gather all call arguments
        callargs = {
            arg: self.call_args_cache[user_id][arg][0] for arg in self.sub_topics.values() # handle scalar args
        } | {
            arg: self.call_args_cache[user_id][arg] for arg in self.sub_topics_queued.values() # handle list-valued args
        }
        if self.user_id:
            callargs['user_id'] = user_id   # add user id, if neccessary
        if self.timestamps:
            callargs['timestamps'] = dict(self.timestamp_cache[user_id])   # add timestamps, if neccessary
        # then, reset arg buffers
        self.call_args_cache[user_id] = defaultdict(lambda: list())
        self.timestamp_cache[user_id] = defaultdict(lambda: list())

        # execute function            
        result = self.fn(other, **callargs)
        if isawaitable(result): # handle coroutine case
            # print("WAIT FOR RESULT")
            result = await result
        return result

    def publish(self, other, topic: str, value: Any, user_id: int):
        print("PUBLISHING RESULT TO", topic + self.domain_suffix)
        serialized = _serialize(topic, value)
        serialized["user_id"] = user_id
        other._component._session.publish(topic + self.domain_suffix, **serialized)

    async def receive(self, other, user_id: int = 0, **kwargs):
        """
        Called whenever a value to a subscribed topic is received.
        Will store all received values since the last function call if topic is subscribed to in queued mode, else will only remember last received value.
        Once at least 1 value is available for each function argument, will call the function and reset the value buffers.
        """
        print(f" --- RECV FOR {self.fn_name}", other, user_id, kwargs)
        self.append_values(user_id, **kwargs)
        if self.ready_for_call(user_id):
            # function has >= 1 values for each argument -> call
            result = await self.call(other=other, user_id=user_id, **kwargs)
            # publish results, if applicable
            for pub_topic in self.pub_topics:
                if pub_topic in result:
                    self.publish(other=other, topic=pub_topic, value=result[pub_topic], user_id=user_id)
                else:
                    msg = f"Function {self.fn} tried to publish to a topic not decared in the decorator and thus will not be published: {pub_topic}"
                    warnings.warn(msg)
                    
            if not isinstance(result, type(None)) and len(set(result.keys()).difference(self.pub_topics)) > 0:
                msg = f"Function {self.fn} published only to a subset of topics, potentially missing topics: {set(result.keys()).difference(self.pub_topics)}"
                warnings.warn(msg)
  

class Service:
    def __init__(self, identifier: str, domain: Union[str, Domain] = "", transports: str = "ws://localhost:8080/ws", realm="adviser") -> None:
        """
        Args:
            identifier: A unique name for the service instance.
                        Used for service control, debugging and remote connections.
            domain: A domain name or domain instance.
                    Appends the domain name as a suffix to all topics inside this service instance.
                    Since topics are prefix-matched, leaving this string empty will result in catching messages from all domains.
            transports: Adress of the dialog system (router).
                        If this service runs on the same machine as the dialog system, usually nothing has to be changed here.
                        If this service runs on a remote machine, this adress should be changed to the adress of the machine running the dialog system.
            realm: Seperation of multiple dialog system instances.
                   Choose a unique string shared between the dialog system instance and all service instances per dialog system instance.
        """
        self.domain = domain
        self._domain_suffix = domain if isinstance(domain, str) else f".{domain.get_domain_name()}"
        self._identifier = identifier
        self._component = Component(transports=transports, realm=realm)
        self._component.on_connect(self._onConnect)
        #self._component.on_connectfailure # TODO
        self._component.on_disconnect(self._onDisconnect)
        self._component.on_join(self._onJoin)
        self._component.on_leave(self._onLeave)
        # self._component.on_ready # TODO

    def _init_pubsub(self, session): 
        """ Search for all functions decorated with the `PublishSubscribe` decorator and call the setup methods for them """
        for func_name in dir(self):
            func_inst = getattr(self, func_name)
            if hasattr(func_inst, "_pubsub"):
                # found decorated publisher / subscriber function -> setup sockets and listeners
                delegate: _ServiceFunctionDelegate = func_inst._delegate
                delegate.set_domain_suffix(self._domain_suffix)
                # subscribe to all topics at central dispatcher
                for topic in set(delegate.sub_topics.keys()).union(delegate.sub_topics_queued.keys()):
                    res = session.subscribe(partial(delegate.receive, other=self), topic=topic, options=SubscribeOptions("prefix"))
                    print("Subscribung to", topic)
    
    async def _setup_ctrl_msg_channel(self, session):
        """
        Setup control channels with the dialog system, including start / stop messages and service registration with the dialog system.
        """
        await session.register(self._on_dialog_start,f"{ControlChannelMessages.DIALOG_START}.{self._identifier}")
        await session.register(self._on_dialog_end, f"{ControlChannelMessages.DIALOG_END}.{self._identifier}")
        await session.register(self._register_service, f'dialogsystem.register.{self._identifier}')
        # print("registered")
        # print("Procedure name:", f'dialogsystem.register.{self._identifier}')

    async def _on_dialog_start(self, user_id: int) -> bool:
        print("DIALOG START:", self._identifier)
        await self.on_dialog_start(user_id)
        return True

    async def _on_dialog_end(self, user_id: int) -> bool:
        await self.on_dialog_end(user_id)
        return False

    def _register_service(self) -> bool:
        print("Call REGISTER function inside service")
        return True

    async def on_dialog_start(self, user_id: int):
        print("Dialog Start", user_id)
        pass

    async def on_dialog_end(self, user_id: int):
        print("Dialog end", user_id)
        pass
        
    def _onConnect(self, session, details):
        print("transport connected")

    def _onChallenge(self, session, challenge):
        print("authentication challenge received")

    async def _onJoin(self, session, details):
        """
        Triggered once the service component is registered with the router.
        At this point, we have a valid transport and can start subsribing to topics.
        """
        # print("JOINING", self._identifier)
        await self._setup_ctrl_msg_channel(session)
        self._init_pubsub(session)

    def _onLeave(self, session, details):
        print("session left")

    def _onDisconnect(self):
        print("transport disconnected")

    def run(self):
        """
        Start the remote service (blocking).
        Use only if this service is to be used on a remote machine / in a different process (w.r.t. the dialog system)
        """
        run(self._component)




def PublishSubscribe(sub_topics: Union[Dict[str, str], Iterable[str]] = [], queued_sub_topics: Union[Dict[str, str], Iterable[str]] = [], pub_topics: Union[Dict[str, str], Iterable[str]] = [], user_id: bool = False, timestamps: bool = False):
    """
    Args:
        sub_topics: A ``dict`` (or ``str``) defining the mapping of topic to function argument.
                    The function will be called once at least one message for all topics in ``sub_topics`` was received,
                    using the last received argument per topic only and discarding the previous messages.

                    * In case of a ``str``, it is assumed that each topic matches one function argument exactly, e.g.::

                        sub_topics = ["topicA", "topicB"] 

                      requires the function declaration to look like this::

                        def fn(self, topicA, topicB)

                    * In case of a ``dict``, this mapping can be customized: E.g.::

                        {"topicA": "arg1", "topicB": "arg2"}

                      allows for the function declaration to look like this::

                        def fn(self, arg1, arg2)

                      This also allows to map different topics to the same function argument::

                        {"topicA": "arg1", "topicB": "arg1"}

                      with function definition::

                        def fn(self, arg1)

                      , thereby changing the function call pattern (the function call will not be stalled until both a message for ``topicA`` and ``topicB`` was received,
                      but instead called each time either ``topicA`` or ``topicB`` will receive a message)
        queued_sub_topics: The same as ``sub_topics``, except that no messages will be discraded.
                           NOTE: Arguments mapped to topics in ``queued_sub_topics`` will receive a ``list`` of values instead of a single value each call.
                           This list will contain all messages received for the corresponding topic in order of reception.
                           NOTE: Both, ``sub_topics`` and ``queued_sub_topics`` arguments can be combined in the same function!
        pub_topics: In case of a ``list``, the service's domain will be automatically appended to the topic, e.g. ``topicA`` -> ``topicA.my_service_instance_domain``.
                    In case of a ``dict``, the service's domain will not be appended to the topic.
                    The second case allows more flexibility, e.g. to choose the specific domain at runtime (like a domain tracker) or to erase it.
        user_id: If true, your decorated function has to include an argument called `user_id`
        timestamps: If true, your decorated function has to include an argument called `timestamps`.
                            This argument will receive a dictionary with the mapping: argument (str) -> List[datetime]  (one timestamp for each value received for the queued sub-topics, or a single list value for non-queued topics)
    """
    def wrapper(func):
        @functools.wraps(func) # we want to keep the original function signature / details
        async def delegate(self, *args, **kwargs):
            # support for calling function as either coroutine or regular function
            print("DELEGATE CALL", func.__name__, args, kwargs)
            if asyncio.iscoroutinefunction(func):
                print("ASYNC", func.__name__)
                result = await func(self, *args, **kwargs)
            else:
                print("NON-ASYNC", func.__name__)
                result = func(self, *args, **kwargs)
        
            if result:
                # publish messages
                for pub_topic in pub_topics:
                    if pub_topic in result:
                        serialized = _serialize(pub_topic, result[pub_topic])
                        serialized["user_id"] = user_id
                        self._component._session.publish(pub_topic + self._domain_suffix, **serialized) 
            # return function result in case decorated function was called normally and not triggered by a subscription event


            return result

        # declare function as publish / subscribe functions and attach the respective topics
        delegate._pubsub = True
        delegate._delegate = _ServiceFunctionDelegate(func, sub_topics, queued_sub_topics, pub_topics, user_id, timestamps)
        return delegate

    return wrapper


class RemoteService:
    """
    Encapsultes a Service on a remote machine (assuming a DialogSytstem is runnning on another machine and we know it's IP address).
    
    Use like this in a seperate script on the remote machine:

        ```
        class MyService(Service):
            ... your service code here

        rs = RemoteService(service_cls=MyService, transports="ws://remote.dialogsystem.ip.address:remoteport/ws")
        rs.run()
        ```

    """
    
    def __init__(self, identifier: str):
        self._identifier = identifier
        

   

