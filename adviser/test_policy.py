from typing import Dict, Union, List
from utils.sysact import SysAct
from services.bst.bst import HandcraftedBST
from utils.domain.domain import Domain
from utils.logger import DiasysLogger
from services.service import DialogSystem, PublishSubscribe
from services.service import Service
from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.nlu.nlu import HandcraftedNLU
from services.policy import HandcraftedPolicy
from services.hci.console import ConsoleInput, ConsoleOutput
from services.domain_tracker.domain_tracker import DomainTracker
from services.hci.gui import GUIServer

import tornado
import tornado.websocket
import json
from utils.topics import Topic


class PassThroughNLG(Service):
    def __init__(self, domain: Union[str, Domain] = "", debug_logger: DiasysLogger = None):
        # super().__init__(domain, debug_logger)
        Service.__init__(self, domain=domain)

    @PublishSubscribe(sub_topics=["sys_act"], pub_topics=["sys_utterance"])
    def pass_on_output(self, sys_act) -> dict(sys_utterance=str):
        print(f"SYS ACTS: {sys_act}")
        return {"sys_utterance": str(sys_act)}

if __name__ == '__main__':
    domain = JSONLookupDomain(name='superhero',
                                json_ontology_file='resources/ontologies/superhero.json', 
                                sqllite_db_file='resources/databases/superhero.db', 
                                display_name='Superhero Domain')
    domain_tracker = DomainTracker(domains=[domain])
    # user_in = ConsoleInput(domain=domain)
    gui_service = GUIServer()
    nlu = HandcraftedNLU(domain=domain)
    bst = HandcraftedBST(domain=domain)
    policy = HandcraftedPolicy(domain=domain)
    nlg = PassThroughNLG(domain=domain)
    # user_out = ConsoleOutput(domain=domain)

    ds = DialogSystem(services=[domain_tracker,nlu, bst, policy, nlg, gui_service])
    error_free = ds.is_error_free_messaging_pipeline()
    if not error_free:
        ds.print_inconsistencies()
        ds.draw_system_graph()
  
    class SimpleWebSocket(tornado.websocket.WebSocketHandler):
        def open(self, *args):
            gui_service.websocket = self
    
        def on_message(self, message):
            data = json.loads(message)
            # check token validity
            topic = data['topic']
            if topic == 'start_dialog':
                # dialog start is triggered from web ui here
                ds._start_dialog({"gen_user_utterance": ""})
            elif topic == 'gen_user_utterance':
                gui_service.user_utterance(message=data['msg'])

        def check_origin(self, *args, **kwargs):
            # allow cross-origin
            return True

    app = tornado.web.Application([ (r"/ws", SimpleWebSocket)])
    app.listen(21512)
    tornado.ioloop.IOLoop.current().start()

    
    # run a single dialog
    ds.run_dialog({'gen_user_utterance': ""})
    # free resources
    ds.shutdown()

