# Purpose:
The `rl` folder contains code related to creating/training a reinforcement learning (RL) agent for dialog policy learning

# File Descriptions:
* `models`: A folder for trained RL policy models
* `dqn.py`: Different Deep-Q Network architectures: DQN and Dueling DQN
* `dqnpolicy.py`: Concrete DQN-based policy (implementation of the RLPolicy`-interface) with options to configure as DQN, Dueling DQN, Double DQN or an arbitrary combination of those.
* `experience_buffer`: Interface for off-policy experience buffers. Contains concrete implementations for random uniform and prioritized buffers.
* `policy_rl.py`: Base class for creating RL-based policies. Includes a lot of utilities (e.g. automatic beliefstate to state-vector conversions, entity querying, action space, ...). Inherit from this class to create a concrete RL-based policy (for an example, have a look at `dqnpolicy.py`)
* `train_dqnpolicy.py`: script for training an DQN-policy (from `dqnpolicy.py`)