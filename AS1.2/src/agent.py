import numpy as np


class Agent:
    def __init__(self, maze_object, policy_object):
        self.maze = maze_object
        self.state = self.maze.start_point
        self.policy = policy_object
        if not self.policy.p_matrix:
            self.policy.p_matrix = np.zeros((self.maze.maze_x_size, self.maze.maze_y_size), dtype='U4')
        self.gamma = self.policy.gamma

    def value_iteration(self):
        """
        This function applies 'Value Iteration' on all the states in the maze that the agent
        currently resides in. It creates a new_value_matrix which it then applies at the end
        of each iteration to avoid calculating the wrong values.
        :@return: void
        """
        delta = 0.01
        iter_count = 0
        while delta > 0:
            new_value_matrix = np.zeros((self.maze.maze_y_size, self.maze.maze_x_size))
            state_value_deltas = []
            for row in range(len(self.maze.value_matrix)):
                for col in range(len(self.maze.value_matrix[row])):
                    if not self.maze.terminal_matrix[row][col]:
                        # Calculate value
                        old_value = self.maze.value_matrix[row][col]
                        neighbours = self.maze.get_reachable_states([row, col])
                        new_value = self.value_function([row, col], neighbours)
                        new_value_matrix[row][col] = new_value

                        # Calculate delta
                        new_delta = old_value - new_value
                        if new_delta < 0:
                            new_delta = -new_delta
                        state_value_deltas.append(new_delta)

                        # Update policy matrix with (new) route
                        self.update_policy([row, col], neighbours)
                    else:
                        self.policy.p_matrix[row][col] = "T"

            delta = max(state_value_deltas)
            if delta > 0:  # Update all the old values with the new values for the next iteration if there will be one.
                self.maze.value_matrix = new_value_matrix

            iter_count += 1

        print(f"\nValue Iteration took '{iter_count}' total iterations (including iteration 0).")

    def value_function(self, pos, neighbours):
        """
        This function calculates the value of the state that is on the given position (pos)
        based on the rewards and values of it's neighbouring states and the gamma of the policy.
        This function is GREEDY because it always returns the most optimal value for the agent if
        the neighbouring state isn't a terminal state. Note: 'nb' means neighbour.
        @param pos: list [y, x]
        @param neighbours: nested list
        @return: float
        """
        values = []
        for nb in neighbours:
            nb_value = self.maze.value_matrix[nb[0][0]][nb[0][1]]
            nb_reward = self.maze.reward_matrix[nb[0][0]][nb[0][1]]
            values.append((nb_reward + (self.gamma * nb_value)))

        return max(values)

    def update_policy(self, current_pos, neighbours):
        best = []
        highest_value = 0.
        for nb in neighbours:
            nb_value = self.maze.value_matrix[nb[0][0]][nb[0][1]]
            nb_reward = self.maze.reward_matrix[nb[0][0]][nb[0][1]]
            nb_action_value = nb_value + nb_reward
            if nb_action_value > highest_value:
                best = []
                highest_value = nb_action_value
                best.append(nb)
            elif nb_action_value == highest_value:
                best.append(nb)

        self.policy.p_matrix[current_pos[0]][current_pos[1]] = "".join([i[1] for i in best])

    def act(self):
        """
        This function applies a chosen action to the agent and makes the Agent
        'act' in the maze environment.
        :@return: void
        """
        action = self.policy.select_action(self.maze.action_values)
        self.state = self.maze.step(self.state, action)

    def show_policy(self):
        print("\nAgent policy matrix:")
        for i in self.policy.p_matrix:
            print(i)
