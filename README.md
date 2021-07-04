# Banana Gathering Reinforcement Learning Agent

This repository contains a pytorch based implementation of a deep Q-learning
agent that can be trained to solve a banana gathering task in an [ml-agents](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md) like
environment.

</br>

## Project Details:

</br>
This project uses the Markov Decision Process (MDP) Framework to solve the banana gathering problem. 

In this environment, the agent is tasked with picking up as many yellow bananas as possible while avoiding gathering purple bananas. 
</br>

### Agent State Space:
</br>
The banana gathering agent is able to chose between 4 possible interactions with the environment at every time step:
</br>
0: move forward</br>
1: move backward</br>
2: turn left</br>
3: turn right</br>
</br>

### Agent Observaton Space:
</br>
The banana gathering agent is able to observe an a state vector from the environment composed of 37 elements.

* 2 elements correspond to orthogonal components of the agents velocity in the plane of the collection arena.
* The remaining elements correspond to a ray-casted representation of objects in front of the agent.
</br>
</br>
### Agent Reward Structure and Solution
</br>
The agent receives a reward of +1 for gathering yellow bananas and -1 for collecting purple bananas. The environment is considered solved when the agent is able to acheive a reward score of 13 or greater for 100 consecutive episodes.
</br>
</br>

## Project Dependencies:
</br>

* Ideally, GPU hardware and access to NVIDIA CUDA
    *  This can be facilitated by setting up an nvidia-docker container based on nvidia/cuda 

### Steps:
</br>

1) Setup a python virtual environment. For example:

    ```
    python -m venv banana_env
    source banana_env/bin/activate
    ```

2) Clone dependencies and install into docker environment:

    </br>

   * Open AI GYM

   </br>

    ```
    git clone https://github.com/openai/gym.git
    cd gym
    pip install -e .
    ```
    </br>

   * Udacity Deep Reinforcement Learning

   </br>

    ```
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

3) Download the standalone Unity based training environment for your use case:

    * [Linux, with visuals](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * [Linux, headless, for training](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)

    Be sure to unzip each. There are others available on that server for Windows

    Note, the software in this repository expects these Unity environments to be unzipped to a mounted directory called data. E.g.:
    ```
    /data/Banana_Linux
    /data/Banana_Linux_NoVis
    ```

## Agent Training:

This section describes the steps to training the agent.

The agent can be trained from the command line or a jupyter-notebook.

* After sourcing the python environment the jupyter-notebook can be started from the command line by calling 
```
jupyter-notebook --ip <IP of host> --port <Port of host>
```
* Alternatively training can be executed from the command line by running:
```
python train.py
```

### Results
* After train.py is run, a plot called model.png is written to disk with the score plotted against the episode number for the last run.
* After all cells of Navigate.ipynb is run, a plot is shown in the notebook.
* Both train.py and Navigate.ipynb result in the creation of model.pt which are the saved weights of the neural network after the last episode of training