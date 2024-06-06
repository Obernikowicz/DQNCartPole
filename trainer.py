import random
import math
import torch
import matplotlib.pyplot as plt
from itertools import count

from replay_memory import Transition

class DQNTrainer():
    def __init__(self, device, env, model, memory, eps_start, eps_end, eps_decay, episodes, batch_size):
        self.device = device

        self.env = env

        self.model = model 
        self.memory = memory

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.episodes = episodes
        self.batch_size = batch_size

        self.steps_done = 0
        self.episode_durations = []
    
    def select_action(self, state):
        sample = random.random()

        eps_treshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_treshold:
            with torch.no_grad():
                return self.model.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
    
    def plot_durations(self, show_result=False):
        plt.figure(1)

        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)

        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')

        plt.xlabel('Episode')
        plt.ylabel('Duration')

        plt.plot(durations_t.numpy())

        plt.pause(0.001)

    def train(self):
        plt.ion()

        for _ in range(self.episodes):
            state, _ = self.env.reset()

            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                self.memory.push(state, action, next_state, reward)

                state = next_state

                if len(self.memory) >= self.batch_size:
                    transitions = self.memory.sample(self.batch_size)

                    batch = Transition(*zip(*transitions))

                    loss = self.model.evaluate(batch, self.batch_size)

                    self.model.optimize(loss)
                
                self.model.update_target_network()

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
        
        print('Complete')

        self.plot_durations(show_result=True)

        plt.ioff()
        plt.show()
    
    def save_model(self, path):
        self.model.policy_net.save(path)