from unityagents import UnityEnvironment
from dqn_agent import Agent
from utils import get_datefmt_str

import torch

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
    "MODE": "TRAIN"
}

def train_rl(agent=None, env=None, training_params=TRAINING_PARAMS, n_episodes=1000, max_t=500, eps_start=0.85, eps_end=0.05, eps_decay=0.996):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episod    
    scores_window = deque(maxlen=100)  # last 100 scores
    max_score = 0
    consecutive = 0
    eps = eps_start                    # initialize epsilon
    brain_name = env.brain_names[0]
    print("train")
    #env.close()
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        #print("env reset...")
        #print(f"state: {state}")
        score = 0
        for t in range(max_t):
            #print(f"timestamp: {t}/{max_t}", end="\r")
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
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        max_score = max(scores)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tMax Score: {:.2f}\teps: {:.2f}'.format(
            i_episode, np.mean(scores_window), max_score, eps), end="\r"
        )
        saved_scores = scores
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tMax Score: {:.2f}\teps: {:.2f}'.format(
                i_episode, np.mean(scores_window), max_score, eps), end="\r"
            )
        if np.mean(scores_window)>=13.5:
            consecutive += 1
        else:
            consecutive = 0
            
        if consecutive >= 100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'model.pt')
            break
    return scores

def outer_loop(training_params=TRAINING_PARAMS, model_name='model'):

    env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    dqn_agent=Agent(state_size, action_size, training_params)

    scores = train_rl(agent=dqn_agent, env=env, training_params=training_params)

    score_meta = {}
    score_meta["max"] = np.max(scores)
    score_meta["min"] = np.min(scores)
    score_meta["episodes"] = len(scores)
    score_meta["mean"] = np.mean(scores)
    score_meta["scores"] = scores

    json_data = json.dumps(score_meta)

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