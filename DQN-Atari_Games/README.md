# Deep Q-Network (DQN) for Atari Games

This repository contains an implementation of the Deep Q-Network (DQN) algorithm for playing Atari games. The DQN algorithm, introduced by Mnih et al. in the paper [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), combines Q-learning with deep neural networks to achieve impressive results in a variety of Atari 2600 games.

## Overview
                                                          
### Deep Q-Network (DQN)

The Deep Q-Network is a deep reinforcement learning algorithm that extends Q-learning to handle high-dimensional state spaces. It employs a neural network to approximate the Q-function, which represents the expected cumulative future rewards for taking a specific action in a given state. This allows DQN to learn directly from raw sensory inputs, making it applicable to a wide range of tasks.

### Atari Games

The Atari 2600, a popular home video game console in the late 1970s and early 1980s, featured a diverse collection of games. These games serve as a benchmark for testing the capabilities of reinforcement learning algorithms. Each game in the Atari 2600 suite provides a unique environment with different challenges, making them an ideal testbed for training agents to generalize across a variety of tasks.

## 任务说明

- 参考README.md文件配置运行环境
- 参考课程PPT完成`dqn_atari.py`中`QNetwork`模块代码
- 参考`train.sh`文件执行训练

## 提交说明

提交一份实验报告，提交修改后的`dqn_atari.py`文件以及生成的最后一次吃豆人游戏视频。

## Getting Started

### Prerequisites

To run this project, you will need the following:

- Python 3.x (3.8)
- PyTorch (1.12.1)
- Gym (OpenAI)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/AI-FDU/DQN-Atari_Games.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

```bash
pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1"
```

## Play

```bash
python playdemo.py
```

## Usage

To train and evaluate the DQN agent, follow the steps outlined below:

1. Set up the required dependencies as described in the [Installation](#installation) section.

2. Train the DQN agent:

```bash
sh train.sh
```  
or
```  
python dqn_atari.py --exp-name MsPacman-v5 --capture-video --env-id ALE/MsPacman-v5 --total-timesteps 5000000 --buffer-size 400000 --hf-entity adhisetiawan
```  

If you want to change the game that you train, please edit the game environment name in `train.sh` file.

3. Evaluate the trained agent:

Evaluation process has been done during end of training, but if you want to run evaluation separately, just run `dqn_eval.py` and change the game environment.

## Training

The training process involves the following steps:

1. Preprocess raw game frames to reduce dimensionality.
2. Initialize a deep neural network to approximate the Q-function.
3. Initialize a replay buffer to store experiences.
4. For each episode, perform the following steps:
   - Select an action using an epsilon-greedy policy.
   - Execute the action in the environment and observe the next state, reward, and terminal flag.
   - Store the experience in the replay buffer.
   - Sample a batch of experiences from the replay buffer and perform a Q-learning update step.
   - Update the target Q-network periodically.

## Evaluation

The evaluation process involves testing the trained DQN agent on a specific game. The agent's performance is measured in terms of the average score achieved over a specified number of episodes.

## Results


### Game: `MS PacMan`

Here's a GIF of the agent playing `MS PacMan`:

![Agent Playing](assets/pacman.gif)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgment
This repository inspired by CleanRL
```
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```
