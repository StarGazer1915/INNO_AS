import numpy as np
from random import choice


class Policy:
    def __init__(self, neural_network, optimizer, loss_function, epsilon):
        self.nn = neural_network
        self.opt = optimizer
        self.loss_fn = loss_function
        self.epsilon = epsilon

    def select_action(self, available_actions, act_values):
        """
        Decides the action that the agent is going to take.
        @param available_actions: list
        @param act_values: dict
        @return: ?
        """
        highest = max(act_values)
        best = [act for act in range(len(available_actions)) if act_values[act] >= highest]
        best_choice = choice(best)

        if np.random.random() <= self.epsilon:
            return choice(available_actions)
        else:
            return best_choice

    def decay(self):
        """
        Degrades epsilon over time.
        @return: void
        """
        return
