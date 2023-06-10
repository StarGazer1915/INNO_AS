import numpy as np
from random import choice


class Policy:
    def __init__(self, neural_network, epsilon):
        self.nn = neural_network
        self.epsilon = epsilon

    def select_action(self, available_actions, act_values):
        """
        Decides the action that the agent is going to take.
        @param available_actions: list
        @param act_values: dict
        @return: ?
        """
        all_values = list(act_values.values())
        if all_values.count(all_values[0]) == len(all_values):
            return choice(available_actions)
        else:
            best = []
            highest = max(all_values)
            for act in available_actions:
                if act_values[act] >= highest:
                    best.append(act)

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
