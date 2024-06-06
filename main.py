# This is sample implementation of DQN solving CartPole-v1 task from Gymnasium.
# The code is based on PyTorch DQN tutorial:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# 
# Solution presented in the tutorial has been analysed and rewritten to be object-oriented
# After training test simulation is beeing run and the result is saved as video

import gymnasium as gym
import torch
import argparse

from config import load_config
from dqn import DQN
from trainer import DQNTrainer
from tester import DQNTester
from replay_memory import ReplayMemory

def run(config_path):
    (LR,
    BATCH_SIZE,
    DISCOUNT_FACTOR,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    UPDATE_RATE,
    EPISODES) = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make('CartPole-v1')
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n 

    model = DQN(device, n_observations, n_actions, LR, DISCOUNT_FACTOR, UPDATE_RATE)
    memory = ReplayMemory(capacity=10000)

    trainer = DQNTrainer(device, env, model, memory, EPS_START, EPS_END, EPS_DECAY, EPISODES, BATCH_SIZE)

    trainer.train()

    trainer.save_model('model.pth')

    tester = DQNTester(device, 'model.pth', n_observations, n_actions)

    tester.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    
    args = parser.parse_args()

    run(args.config)
