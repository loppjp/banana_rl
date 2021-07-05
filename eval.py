from unityagents import UnityEnvironment
from dqn_window_agent import Agent
from utils import get_datefmt_str

import torch

import time
import os
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np

saved_scores=[]

TRAINING_PARAMS = {
    "REPLAY_BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 64,
    "GAMMA": 0.95,
    "TAU": 1e-2,
    "LEARNING_RATE": 1e-3,
    "UPDATE_TARGET_NET_STEPS": 4,
    "SEED": int(1234),
    "MODE": "EVAL"
}

def eval_rl(agent=None, model='model.pth', env=None, training_params=TRAINING_PARAMS, n_episodes=1000, max_t=5000, eps_start=0.85, eps_end=0.05, eps_decay=0.996):
    """
    Evaluate the agent in a single episode. Loop over timesteps 
 
    Function derived from:
    udacity.com Deep Reinforcement Learning Nanodegree:
    Part 2, Lesson 2, Deep-Q Networks
    
    Arguments:
        agent - An instance of Agent, implements functions such as step, act, lean
        model - the relative or absolute path to a file containing model weights
        env - A unity agents environnment instance for this training session
        training_params - A dictionary of parameters used for training, see above
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    # load the provided model file
    agent.qnetwork_local.load_state_dict(torch.load(model))

    rewards = []
    score = 0 # score acheived
    max_score = 0
    eps = 0.05 # eval epsilon
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    for t in range(max_t):
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(
            state, 
            action, 
            reward, 
            next_state, 
            done
        )
        state = next_state
        score += reward
        rewards.append(reward)
        if done:
            break 
        max_score = max(max_score, score)
        print('\rIteration {}\tAverage Reward: {:.2f}\tMax Score: {:.2f}'.format(
            t, np.mean(rewards), max_score), end="\r"
        )

        time.sleep(0.01)

    print(f"{os.linesep}total rewards: {np.sum(rewards)}")

    return rewards

def outer_loop(training_params=TRAINING_PARAMS, model_name='model'):

    env = UnityEnvironment(file_name="/data/Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    # instantiate the agent
    dqn_agent=Agent(state_size, action_size, training_params)

    # run an episode for evaluation
    scores = eval_rl(agent=dqn_agent, env=env, training_params=training_params)

    score_meta = {}
    score_meta["max"] = np.max(scores)
    score_meta["min"] = np.min(scores)
    score_meta["episodes"] = len(scores)
    score_meta["mean"] = np.mean(scores)
    score_meta["scores"] = scores

    json_data = json.dumps(score_meta)

    # save evaulation results
    with open(f"{model_name}_{get_datefmt_str()}.json", 'w') as f:
        f.write(json_data)

    # plot the scores
    matplotlib.use('Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'{model_name}_{get_datefmt_str()}.png')

    env.close()

if __name__ == "__main__":

    outer_loop()
