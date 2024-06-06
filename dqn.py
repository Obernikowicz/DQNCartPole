import torch
import torch.nn as nn
import torch.optim as optim 

from qnet import QNet

class DQN():
    def __init__(self, device, n_observations, n_actions, lr, discount_factor, update_rate):
        self.device = device

        self.policy_net = QNet(n_observations, n_actions)
        
        self.target_net = QNet(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)

        self.discount_factor = discount_factor
        self.gamma = self.discount_factor
        self.update_rate = update_rate

    def evaluate(self, batch, batch_size):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device = self.device,
                                      dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        return loss
    
    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.update_rate + target_net_state_dict[key] * (1 - self.update_rate)

        self.target_net.load_state_dict(target_net_state_dict)