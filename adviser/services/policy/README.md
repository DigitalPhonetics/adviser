# Purpose:
The policy folder contains code related to system decision making. Currently this means code related to handcrafted and RL policies for deciding the next system act and code related to a handcrafted policy for deciding what emotion the system will respond with.

# File Descriptions:
* `rl`: a folder containing code necessary to create a reinforcement learning (RL) agent for dialog policy.
* `affective_policy.py`: Defines the `EmotionPolicy` class which maps the user state to a system emotion for output.
* `policy_api.py`: Defines a policy class for handling API domains. This is different from the standard policy, which focuses on finding and entity and asking questions about it, becuase the definition of entity is a bit more fluid in some API domain. For example, in the weather domain, an entity would need to be thought of as a combination of place and time before it makes sense to ask about the weather. This policy is designed to navigate that type of ambiguity.
* `policy_handcrafted.py`: Defines a handcrafted policy class for deciding the next system action.