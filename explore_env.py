from unityagents import UnityEnvironment
import numpy as np

# please do not modify the line below
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# examine change
steps = 5
act = 1

for x in range(0, steps):
    env_info = env.step(act)[brain_name]
next_state = env_info.vector_observations[0]
print(f"Next state: {next_state}")

for x in range(0, steps):
    env_info = env.step(act)[brain_name]
next_state = env_info.vector_observations[0]
print(f"Next state: {next_state}")