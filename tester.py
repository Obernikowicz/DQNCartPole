import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from qnet import QNet

class DQNTester():
    def __init__(self, device, model_path, n_observations, n_actions):
        self.device = device

        self.env = RecordVideo(gym.make('CartPole-v1', render_mode='rgb_array'), './mp4')
        self.env.reset()

        self.model = QNet(n_observations, n_actions).load(model_path)

    def select_action(self, state):
        return self.model(state).max(1).indices.view(1, 1)
    
    def test(self):
        state, info = self.env.reset()

        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        for i in range(1000):
            action = self.select_action(state)

            observation, _, terminated, truncated, _ = self.env.step(action.item())

            if terminated or truncated:
                print(f'Episode finished after {i + 1} timesteps.')
                break

            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.env.close()

