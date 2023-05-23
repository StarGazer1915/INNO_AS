import numpy as np
from random import choice


class Policy:
    def __init__(self, policy_type="", p_matrix=np.array([])):
        self.type = policy_type
        self.p_matrix = p_matrix

    def select_action(self, actions, current_state):
        """
        This function decides the action that the agent is going
        to take within the maze environment.
        @param actions: nested list
        @return: list [y, x]
        """
        if self.type.lower() in "random":
            return choice(actions)
        elif self.type.lower() == "on-policy":
            return self.p_matrix[current_state[0]][current_state[1]]
