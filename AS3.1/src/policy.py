import numpy as np
from random import choice
from torch import nn
from torch.optim import Adam


class Policy:
    def __init__(self, neural_network, learning_rate, epsilon, actions, epsilon_decay):
        self.nn = neural_network
        self.lr = learning_rate
        self.opt = Adam(self.nn.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        self.actions = actions
        self.epsilon_decay = epsilon_decay

    def select_action(self, act_values):
        """
        Decides the action that the agent is going to take.
        @param act_values: dict
        @return: int
        """
        highest = max(act_values)
        best = [act for act in range(len(self.actions)) if act_values[act] >= highest]
        best_choice = choice(best)

        if np.random.random() <= self.epsilon:
            return choice(self.actions)
        else:
            return best_choice

    def decay(self):
        """
        Degrades epsilon over time.
        @return: void
        """
        self.epsilon *= self.epsilon_decay
