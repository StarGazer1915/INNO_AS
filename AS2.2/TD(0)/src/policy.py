import numpy as np


class Policy:
    def __init__(self, env_size=(0, 0), p_matrix=np.array([])):
        self.env_size = env_size
        if len(p_matrix) < 1:
            self.p_matrix = np.zeros((self.env_size[0], self.env_size[1]), dtype="U4")
        else:
            self.p_matrix = p_matrix

    def select_action(self, current_state):
        """
        This function decides the action that the agent is going
        to take within the maze environment.
        @param actions: list
        @param current_state: list [y, x]
        @param q_table: float or dict
        @return: list [y, x]
        """
        return self.p_matrix[current_state[0]][current_state[1]]