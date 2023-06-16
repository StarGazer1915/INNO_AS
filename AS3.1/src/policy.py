import numpy as np
from random import choice


class Policy:
    def __init__(self, neural_network, learning_rate, epsilon, actions, epsilon_decay):
        self.nn = neural_network
        self.opt = None
        self.loss_fn = None
        self.lr = learning_rate
        self.epsilon = epsilon
        self.actions = actions
        self.epsilon_decay = epsilon_decay

    def select_action(self, act_values):
        """
        Decides the action that the agent is going to take.
        @param act_values: dict
        @return: int
        """
        best_choice = choice([act for act in self.actions if act_values[act] >= max(act_values)])
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
