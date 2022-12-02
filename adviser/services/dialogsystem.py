import asyncio
from typing import Any, Dict, Iterable, List, Union
from autobahn.asyncio.component import Component, run
from autobahn.wamp import SubscribeOptions

from services.service import ControlChannelMessages, RemoteService, Service

# Later TODO load balancer (manage multiple instances of the same service, distribute calls between them)?
# TODO system graph
# TODO error messages
# TODO start / end messages
# TODO try console input (is it cancellable?)
# TODO keep a list of RPC calls currently not reachable during registration of services, print this list with every retry, update with every successfull registration

class _ServiceConfiguration:
    def __init__(self, identifier: str, instance: Union[Service, RemoteService], remote: bool) -> None:
        self.identifier = identifier
        self.instance = instance
        self.remote = remote
        self.connected = False
        self.active = False

class DialogSystem:
    # TODO start router 
    def __init__(self, services: List[Union[Service, RemoteService]], transports: str = "ws://localhost:8080/ws", realm="adviser") -> None:
        # organize all services (remote & local)
        self.components = {service._identifier: _ServiceConfiguration(service._identifier, service, isinstance(service, RemoteService)) for service in services}

        self._ctrl_component = Component(transports=transports, realm=realm)
        self._ctrl_component.on_connect(self._onConnect)
        #self._component.on_connectfailure # TODO
        self._ctrl_component.on_disconnect(self._onDisconnect)
        self._ctrl_component.on_join(self._onJoin)
        self._ctrl_component.on_leave(self._onLeave)
        # self._component.on_ready # TODO

        self._all_connected = asyncio.Condition()
        self._all_active = asyncio.Condition()

        # stats
        self.dialog_counter = 0

    async def _publish(self, topic: str, value: Any):
     
        async with self._all_active:
            await self._all_active.wait_for(lambda: all([self.components[component].active for component in self.components]))

        self._ctrl_component._session.publish(topic, **{topic: value})

    async def _startup(self, tasks):
        # print(tasks)
        await asyncio.gather(*tasks)

    async def _on_dialog_state_changed(self, user_id: int, **kwargs):
        """ Wait for all services to reply with an ACK to the DIALOG_END message  """
        # for key in kwargs:
        #     if kwargs[key] == False:
        #         return

        async with self._all_connected:
            await self._all_connected.wait_for(lambda: all([self.components[component].connected for component in self.components]))

        for component in self.components:
            self.components[component].active = await self._ctrl_component._session.call(f"{ControlChannelMessages.DIALOG_START}.{self.components[component].identifier}", user_id=user_id)
            async with self._all_active:
                self._all_active.notify_all()

    def run(self, start_messages: Dict[str, Any] = {ControlChannelMessages.DIALOG_START: True}):
        """
        Run the dialog system.
        WARNING: This is a blocking function call!
        (no functions after calling this function will be executed until a SHUTDOWN message is sent or the process is terminated from the outside).

        Args:
            start_messages: None (dialog starts have to be triggered by inputs) or a mapping: topic -> value (e.g. DIALOG_START: True, USER_UTTERANCE: "")
        """
        # TODO fire start_messages: create a coroutine task for publishing them, add task to event look
        tasks = [self._on_dialog_state_changed(user_id=0, changed=True)]
        for msg in start_messages:
            tasks.append(self._publish(msg, start_messages[msg]))
            # print("Created startup msg task for", msg)
        # print(f"Starting {len([component.instance._component for component in self.components if not component.remote])} components...")
        run([self.components[component].instance._component for component in self.components if not self.components[component].remote] + [self._ctrl_component], start_loop=False)
        # print("STARTING LOOP", tasks)
        if len(tasks) > 0:
            asyncio.get_event_loop().create_task(self._startup(tasks))
        asyncio.get_event_loop().run_forever()

    async def _register_component(self, identifier: str, sub_topics: Dict[str, Iterable[str]], sub_topics_queued: Dict[str, Iterable[str]], pub_topics: Dict[str, Iterable[str]]):
        """
        Register a local or remote services via RPC.

        Args:
            identifier: the realm-unique identifier
            sub_topics: a mapping of the topics subscribed to by the component to be registered: function name -> topic names
            sub_topics: a mapping of the queued topics subscribed to by the component to be registered: function name -> topic names
            sub_topics: a mapping of the topics published by the component to be registered: function name -> topic names
        """
        # TODO use this information to draw a system graph / debugging help
        print("trying to regiser component", identifier, "with services", sub_topics, sub_topics_queued, pub_topics)
        self.components[identifier].connected = True
        self.components[identifier].active = False
        async with self._all_connected:
            self._all_connected.notify_all()
       
    def draw_dialog_graph(self):
        # TODO
        pass

    def check_system(self):
        # TODO find errors / warnings for connections between topics
        pass 
    
    def _onConnect(self, session, details):
        print("DS: transport connected")

    def _onChallenge(self, session, challenge):
        print("DS: authentication challenge received")

    async def _onJoin(self, session, details):
        """
        Triggered once the service component is registered with the router.
        At this point, we have a valid transport and can start subsribing to topics.
        """
        print("init pubsub system")
        # Control channel messaging system setup
        # print("INIT CTRL CHANNELS")
        # TODO self._ctrl_component.subscribe(ControlChannelMessages.DIALOGSYSTEM_SHUTDOWN)
        self._ctrl_component._session.subscribe(self._on_dialog_state_changed, topic=ControlChannelMessages.DIALOG_START, options=SubscribeOptions("prefix"))
        self._ctrl_component._session.subscribe(self._on_dialog_state_changed, topic=ControlChannelMessages.DIALOG_END, options=SubscribeOptions("prefix"))
        await self._ctrl_component._session.register(self._register_component, ControlChannelMessages._SERVICE_REGISTER)
        print("Done init pubsub system")

    def _onLeave(self, session, details):
        print("DS: session left")

    def _onDisconnect(self):
        print("DS: transport disconnected")

    # def start(self, num_dialogs: int = 1):
    #     """
    #     This is a blocking function!
    #     After calling, it runs as many dialogs as specified before any code written after the call to this function will be executed.

    #     Args:
    #         num_dialogs (int): If > 0, system will stop after `num_dialogs` dialogs. 
    #                            If < 0, system will run in an infinite loop (until the process is stopped from outside). Useful e.g. for hosting.
    #     """
    #     assert num_dialogs != 0, "Number of dialogs should be > 0 (fixed amount) or < 1 (infinite loop), but got 0"
    #     run(self.components)
    #     # TODO check that all components are running and connected without error
    #     # TODO manage start dialog / end dialog events

    # def start_dialog(self, )
