# Purpose:
This folder contains resources required to run an Adviser 2 dialog system. These include the regexes/templates used by the NLU/NLG to understand/generate natural langauge utterances and the ontologies and databases which define the entities, slots, and values used for a specific domain. Additionally, models for speech recognition and synthesis, emotion recognition, backchannel prediction, and facial landmark detection are contained in the models folder.

# File Descriptions:
* `databases`: Folder containing SQLite databases which define the entities for each domain
* `models`: Folder containing trained models for machine learning tasks, currently there are models for: speech recognition and synthesis, emotion recognition, backchannel prediction, facial landmark detection
* `nlg_templates`: A folder containing the templates for turning system actions to natural language output for each domain
* `nlu_regexes`: A folder containing the regexes for turning natural language input into system actions for each domain
* `ontologies`: Folder containing JSON files which define the slots and values and which of these are user/system requestable/informable for each domain
