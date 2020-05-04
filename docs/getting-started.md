# Getting Started

## Installing the Adviser Toolkit

Get the [code](https://github.com/DigitalPhonetics/adviser) and follow the install instructions.

## Testing Your Installation

1. Open a terminal
2. Activate your virtual environment for Adviser (as created in the install instructions)
3. Navigate to the `adviser` folder containing the `run_chat.py` file
4. Execute `python run_chat.py mensa`
    - This will start a new dialog with a domain about the food plan of the University of Stuttgart's dining hall
    - Try chatting with the system e.g. type something like
        - `What's the vegetarian main dish today`
    - In case you encounter any errors, try to make sure your setup is correct.
      If the problem persists, feel free to write an email to [support](mailto:adviser-support@ims.uni-stuttgart.de),
      providing the full stack trace and, if possible, the dialog turn history.

## Creating and Running Your Own Dialog System

To setup your own text-based dialog system (in this example, for a domain about lecturer information from the IMS Institute at the Unviersity of Stuttgart), it's as simple as creating a new file in the 'adviser' folder (where the `run_chat.py` file is located),
naming it e.g. `mydiasys.py`, and adding the following content:

```python
import sys
import os
from typing import List

from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.service import Service, PublishSubscribe, DialogSystem
from services.domain_tracker import DomainTracker
from services.nlu.nlu import HandcraftedNLU
from services.nlg.nlg import HandcraftedNLG
from services.bst import HandcraftedBST
from services.policy import HandcraftedPolicy
from services.hci import ConsoleInput, ConsoleOutput
from utils.logger import DiasysLogger, LogLevel

# create modules
lecturers_domain = JSONLookupDomain('ImsLecturers')
nlu = HandcraftedNLU(domain=lecturers_domain)
bst = HandcraftedBST(domain=lecturers_domain)
policy = HandcraftedPolicy(domain=lecturers_domain)
nlg = HandcraftedNLG(domain=lecturers_domain)
d_tracker = DomainTracker(domains=[lecturers_domain])

# Input modules (just allow access to terminal for text based dialog)
user_in = ConsoleInput(domain="")
user_out = ConsoleOutput(domain="")

logger = DiasysLogger(console_log_lvl=LogLevel.DIALOGS)
ds = DialogSystem(services=[d_tracker, user_in, user_out, nlu, bst, policy, nlg], debug_logger=logger)

error_free = ds.is_error_free_messaging_pipeline()
if not error_free:
    ds.print_inconsistencies()

ds.draw_system_graph(name="testgraph")

# start dialog
for _ in range(1):
    ds.run_dialog({'gen_user_utterance': ""})
ds.shutdown()
```

To run this code, execute `python mydiasys.py`.
You can try e.g. utterances like `I want information about a Digital Phonetics lecturer` and continue the dialog from there.