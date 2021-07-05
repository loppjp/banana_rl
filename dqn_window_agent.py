import random

import torch
import numpy as np

from dq_network import QNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """
    A deep Q-Learning agent that stores the previous observation state vector.
    When training and inferencing, concatenate the previous and current state
    vector
    """

    def __init__(self, state_size, action_size, training_params):
        """
        Construct the agent

        Arguments:
            state_size: An integer to provide the size of the observation
                        state vector
            action_size: An integer to provide the size of the action vector
            training_params: a dictionary of parameters for training
        """
        
        # double the size of the state vector since the previous and current 
        # observations are twice as large as just one
        self.state_size = state_size * 2

        self.action_size = action_size
        self.seed = random.seed(training_params["SEED"])
        self.training_params = training_params

        # Q-Network
        self.qnetwork_local = QNetwork(
            self.state_size, 
            action_size, 
            self.training_params["SEED"]
        ).to(device)

        self.qnetwork_target = QNetwork(
            self.state_size, 
            action_size, 
            self.training_params["SEED"]
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.qnetwork_local.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )

        self.loss = torch.nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, 
            self.training_params["REPLAY_BUFFER_SIZE"], 
            self.training_params["BATCH_SIZE"], 
            self.seed
        )
        
        # initialize the previous state to zeros
        self.last_state = np.zeros(state_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(
            np.concatenate((state, self.last_state), axis=None), 
            action, 
            reward, 
            np.concatenate((next_state, state), axis=None), 
            done
        )
        
        # save the last state for the next step in the future
        self.last_state = state
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.training_params["UPDATE_TARGET_NET_STEPS"]
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.training_params["BATCH_SIZE"]:
                experiences = self.memory.sample()
                self.learn(experiences, self.training_params["GAMMA"])

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        
        Arguments:
            state: a numpy array providing the current state
            eps: epsilon, float value, for epsilon-greedy action selection
        """
        state = torch.from_numpy(np.concatenate((state, self.last_state), axis=None)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Train on a batch of trajectories and update the target network

        Arguments:
            experiences - tuple of torch tensors representing trajectories
            gamma - floating point discounting factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.training_params["GAMMA"] * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = self.loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.training_params["TAU"])                     

    def soft_update(self, local_model, target_model, tau):
        """
        Use tau to determine to what extent to update target network

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            local_model - pytorch neural network model. used for actions
            target_model - pytorch neural network model. used for training
            tau - ratio by which to update target from local
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
