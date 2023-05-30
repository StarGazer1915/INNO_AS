import numpy as np
from random import choice


class Policy:
    def __init__(self, policy_type="", env_size=(0, 0), p_matrix=np.array([]), epsilon=0.):
        self.type = policy_type
        self.env_size = env_size
        if not p_matrix or len(p_matrix) < 1:
            self.p_matrix = np.zeros((self.env_size[0], self.env_size[1]), dtype="U4")
        else:
            self.p_matrix = p_matrix
        self.epsilon = epsilon

    def select_action(self, actions, current_state, current_value):
        """
        This function decides the action that the agent is going
        to take within the maze environment.
        @param actions: nested list
        @param current_state: list [y, x]
        @param current_value: float or dict
        @return: list [y, x]
        """
        actions = np.array(actions)  # array(['L', 'R', 'U', 'D'])
        if self.type.lower() in "random":
            return np.random.choice(actions, 1)

        elif self.type.lower() == "td(0)":
            return self.p_matrix[current_state[0]][current_state[1]]

        elif self.type.lower() == "sarsa":
            all_values = list(current_value.values())
            if all_values.count(all_values[0]) != len(all_values):
                greedy_choice = max(current_value, key=current_value.get)
                chance_threshold = 1.0 - self.epsilon
                chance = (np.random.randint(100, size=1) / 100)[0]
                if chance <= chance_threshold:
                    return greedy_choice
                else:
                    return choice(actions)
            else:
                return choice(list(current_value.keys()))
