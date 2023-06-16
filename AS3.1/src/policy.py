import random
import torch
import math


class Policy:
    def __init__(self, policy_network, device, actions, epsilon_start, epsilon_end, epsilon_decay):
        self.neural_net = policy_network
        self.device = device
        self.actions = actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

    def select_action(self, state):
        """
        Decides the action that the agent is going to take.
        @param state: numpy array
        @return: int
        """
        self.decay()
        self.step_count += 1
        if random.random() > self.epsilon:
            with torch.no_grad():
                return torch.tensor([self.neural_net(state).argmax()], dtype=torch.int64, device=self.device)
        else:
            return torch.tensor([self.actions.sample()], dtype=torch.int64, device=self.device)

    def decay(self):
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(-1. * self.step_count / self.epsilon_decay)
