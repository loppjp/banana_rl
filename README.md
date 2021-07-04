# Banana Gathering Reinforcement Learning Agent

This repository contains a pytorch based implementation of a deep Q-learning
agent that can be trained to solve a banana gathering task in an [ml-agents](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md) like
environment.

</br>

## Project Details:

</br>

This project uses the Markov Decision Process (MDP) Framework to solve the banana gathering problem. 

In this environment, the agent is tasked with picking up as many yellow bananas as possible while avoiding gathering purple bananas. 

### Agent State Space:

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

3) Download the standalone Unity based training environment:

    * [Linux, with visuals](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * [Linux, headless, for training](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)

    Be sure to unzip each. There are others available on that server for Windows
