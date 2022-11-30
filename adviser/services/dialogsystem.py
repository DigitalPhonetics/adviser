import asyncio
from typing import Any, Dict, List, Union
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

        # stats
        self.dialog_counter = 0

    async def _publish(self, topic: str, value: Any):
        while (isinstance(self._ctrl_component.__getattribute__("_session"), type(None)) or isinstance(self._ctrl_component._session, type(None))) or not all([self.components[component].connected for component in self.components]) or not all([self.components[component].active for component in self.components]):
            print("Retry sending to", topic)
            await asyncio.sleep(0.1)
        self._ctrl_component._session.publish(topic, **{topic: value})

    async def _startup(self, tasks):
        # print(tasks)
        await asyncio.gather(*tasks)

    async def _on_dialog_state_changed(self, user_id: int, **kwargs):
        """ Wait for all services to reply with an ACK to the DIALOG_END message  """
        for key in kwargs:
            if kwargs[key] == False:
                return
        while (isinstance(self._ctrl_component.__getattribute__("_session"), type(None)) or isinstance(self._ctrl_component._session, type(None))) or not all([self.components[component].connected for component in self.components]):
            await asyncio.sleep(0.1)
        for component in self.components:
            self.components[component].active = await self._ctrl_component._session.call(f"{ControlChannelMessages.DIALOG_START}.{self.components[component].identifier}", user_id=user_id)

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

    async def _register_component(self, session, component: _ServiceConfiguration):
        """
        Register all local and remote services via RPC.
        This function will retry to call the registration serivce remote procedure associated with component every second.
        """
        # print("trying to connect to component", component.identifier)
        while not component.connected:
            try:
                # print("REGISTER CALL FOR", component.identifier)
                component.connected = await session.call(f'dialogsystem.register.{component.identifier}')
            except:
                # print("retry")
                await asyncio.sleep(0.5)
        print("-> registered component", component.identifier)

    async def _register_components(self, session):
        await asyncio.gather(*[self._register_component(session, self.components[component]) for component in self.components])
        print("-- ALL SYSTEMS CONNECTED -- ")
        # TODO check system 

    async def _init_pubsub(self, session):
        # Control channel messaging system setup
        # print("INIT CTRL CHANNELS")
        # TODO self._ctrl_component.subscribe(ControlChannelMessages.DIALOGSYSTEM_SHUTDOWN)
        self._ctrl_component._session.subscribe(self._on_dialog_state_changed, topic=ControlChannelMessages.DIALOG_START, options=SubscribeOptions("prefix"))
        self._ctrl_component._session.subscribe(self._on_dialog_state_changed, topic=ControlChannelMessages.DIALOG_END, options=SubscribeOptions("prefix"))
            
    def draw_dialog_graph(self):
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
        print("registering components...")
        await self._register_components(session)
        print("init pubsub system")
        await self._init_pubsub(session)
        # print("pubsub running")

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
