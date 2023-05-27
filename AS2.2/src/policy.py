import numpy as np


class Policy:
    def __init__(self, policy_type="", p_matrix=np.array([]), epsilon=0.):
        self.type = policy_type
        self.p_matrix = p_matrix
        self.epsilon = epsilon

    def select_action(self, actions, current_state):
        """
        This function decides the action that the agent is going
        to take within the maze environment.
        @param actions: nested list
        @param current_state: list [y, x]
        @return: list [y, x]
        """
        actions = np.array(actions)  # array(['L', 'R', 'U', 'D'])
        if self.type.lower() in "random":
            return np.random.choice(actions, 1)

        elif self.type.lower() == "on-policy":
            return self.p_matrix[current_state[0]][current_state[1]]

        elif self.type.lower() == "on-policy + epsilon":
            p_action = self.p_matrix[current_state[0]][current_state[1]]
            e_chance = self.epsilon / len(actions)
            rest_chance = 1 - self.epsilon
            probabilities = np.zeros((len(actions))) + e_chance
            if len(p_action) == 1:  # If policy only gives a single direction
                probabilities[actions == p_action] += rest_chance
                return np.random.choice(actions, 1, p=probabilities)[0]
            elif len(p_action) > 1:  # If policy gives multiple directions
                for act in p_action:
                    probabilities[actions == act] += rest_chance / len(p_action)
                return np.random.choice(actions, 1, p=probabilities)[0]
            else:  # If policy has no existing direction to give
                return np.random.choice(actions, 1)
