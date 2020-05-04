# Purpose:
The service folder holds the core of the system functionality. Each folder contains implementations of services which can be used to construct a dialog system.

# File Descriptions:
* `service.py`: Describes the service class, which provides the communication backbone for services to interact with one another and form a dialog system
* `backchannel`: A folder for code related to determining if/what kind of backchannel is appropriate given a user utterance
* `bst`: A folder for code related to the Belief State Tracker (BST); which is responsible for providing a memory of what information the user has contributed to a conversation
* `domain_tracker`: A folder for code related to determining which domain should be active at a given time in a dialog
* `emotion`: A folder for code related to determining the user's current emotional state
* `engagement`: A folder for code related to determining a user's current engagement level
* `hci`: A folder for code which determines how a user can interact with the system (text input/output, speech input/output, a GUI, etc.)
* `nlg`: A folder for code related to Natural Langugage Generation (NLG) which converts the semantic representation of system actions to a natural language one
* `nlu`: A folder for code related to  Natural Language Understanding (NLU), which maps a user utterance to a semantic representation
* `policy`: A folder for code related to the deciding the next system action or the system emotional response
* `simulator`: A folder for code used to stand in for human users when training the dialog system
* `stats`: A folder for code related to measuring the performance of various services, right now only for policy evaluation
* `ust`: A folder for code related to User State Tracking (UST), keeping track of the user's emotion engagement level