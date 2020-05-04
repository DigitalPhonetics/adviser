# Purpose:
The nlu folder contains code related to natural language understanding. Currently this is only a handcrafted natural language understanding module, but could be expanded to include machine learning approaches.

# File Descriptions:
`nlu.py`: The natural language understanding (NLU) service currently reads in a regex file and uses this to determine the semantic represention of the user's utterance. Converting string input to one or more `UserAct`s.