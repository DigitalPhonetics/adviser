# Purpose:
The simulator folder provides code for simulators which can be used to test/train services in the dialog system when it isn't possible or practical to do so with a real user.

# File Descriptions:
* `emotion_simulator.py`: A class for generating user emtions and engagement levels. Currently only randomly generates or outputs a given fixed value. Used to the affective policy and affective NLG.
* `goal.py`: Defines the `Goal` class which is needed by the user simulator to provide constraints and requests the simulated user should give.
* `simulator.py`: Defines the `HandcraftedUserSimulator` and `Agenda` classes which are used to simulate a user for training the RL policy
* `usermodel.cfg`: A configuration file where properties of the user simulator can be fine-tuned.