import sys
import os

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.join("..", (os.path.abspath(__file__)))))

sys.path.append(get_root_dir())
print(sys.path)


# import domain class and logger
from utils.logger import DiasysLogger, LogLevel
from utils.domain.jsonlookupdomain import JSONLookupDomain

# import services needed for the dialog system
from services.domain_tracker import DomainTracker 
from services.hci import ConsoleInput
from services.nlu import HandcraftedNLU
from services.bst import HandcraftedBST
from services.policy import HandcraftedPolicy
from services.nlg import HandcraftedNLG
from services.hci import ConsoleOutput

# import dialog system class
from services.service import DialogSystem



# 1. Create a JSONLookupDomain object for the "superhero" domain

# 2. For each service, create an object, don't forget to pass the correct domain as an argument
#    Refer back to the last section of the tutorial if you have trouble

# 3. Create a dialog system object and register all the necessary services to it

# 4. Add code to run your dialog system


