import numpy as np
from random import choice


class Policy:
    def __init__(self, policy_type="", env_size=(0, 0), p_matrix=np.array([]), epsilon=0.):
        self.type = policy_type
        self.env_size = env_size
        if len(p_matrix) < 1:
            self.p_matrix = np.zeros((self.env_size[0], self.env_size[1]), dtype="U4")
        else:
            self.p_matrix = p_matrix
        self.epsilon = epsilon

    def select_action(self, actions, current_state, q_table=None):
        """
        This function decides the action that the agent is going
        to take within the maze environment.
        @param actions: list
        @param current_state: list [y, x]
        @param q_table: float or dict
        @return: list [y, x]
        """
        actions = np.array(actions)  # array(['L', 'R', 'U', 'D'])
        if self.type.lower() in "random":
            return np.random.choice(actions, 1)

        elif self.type.lower() == "td(0)":
            return self.p_matrix[current_state[0]][current_state[1]]

        elif self.type.lower() == "sarsa":
            all_values = list(q_table.values())
            if all_values.count(all_values[0]) == len(all_values):
                return choice(actions)
            else:
                best = []
                highest = max(all_values)
                for act in actions:
                    if q_table[act] >= highest:
                        best.append(act)

                best_choice = choice(best)
                if np.random.random() >= self.epsilon:
                    return choice(actions)
                else:
                    return best_choice
