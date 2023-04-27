import numpy as np


class Agent:
    def __init__(self, maze_object, policy_object):
        self.maze = maze_object
        self.state = self.maze.start_point
        self.policy = policy_object
        self.possible_actions = list(self.maze.actions.values())
        self.gamma = self.policy.gamma
        self.ended = False

    def get_reachable_states(self, state_coordinate):
        """
        This function returns the coordinates (indexes) of the reachable states
        based on the possible actions that are available to the agent. It looks for
        states by checking in what states you find yourself if you execute each of
        the available actions. It then returns the coordinates of these states.
        @param state_coordinate: list [y, x]
        @return: nested list of coordinates
        """
        neighbouring_state_coordinates = []
        for act in self.possible_actions:
            nb = [state_coordinate[0] + act[0], state_coordinate[1] + act[1]]
            try:
                check_if_inside_maze = self.maze.environment[nb[0]][nb[1]]
                if nb[0] >= 0 and nb[1] >= 0:
                    neighbouring_state_coordinates.append(nb)
            except IndexError:
                pass

        return neighbouring_state_coordinates

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
            for row in range(len(self.maze.environment)):
                for col in range(len(self.maze.environment[row])):
                    old_value = self.maze.environment[row][col].value
                    new_value = self.value_function(self.maze.environment[row][col].state_coordinate)
                    new_delta = old_value - new_value
                    if new_delta < 0:
                        new_delta = -new_delta
                    state_value_deltas.append(new_delta)
                    new_value_matrix[row][col] = new_value

            delta = max(state_value_deltas)
            if delta > 0:  # Update all the old values with the new values for the next iteration (if there is one).
                for row in range(len(self.maze.environment)):
                    for col in range(len(self.maze.environment[row])):
                        self.maze.environment[row][col].value = new_value_matrix[row][col]

            iter_count += 1

        print(f"\nThe result below took '{iter_count}' total iterations (including iteration 0).")
        self.maze.print_detailed_result_matrix()

    def value_function(self, pos):
        """
        This function calculates the value of the state that is on the given position (pos)
        based on the rewards and values of it's neighbouring states and the gamma of the policy.
        This function is GREEDY because it always returns the most optimal value for the agent if
        the neighbouring state isn't a terminal state. Note: 'nb' means neighbour.
        @param pos: list [y, x]
        @return: float
        """
        if not self.maze.environment[pos[0]][pos[1]].terminal:
            values = []
            for nb in self.get_reachable_states(pos):
                nb_value = self.maze.environment[nb[0]][nb[1]].value
                nb_reward = self.maze.environment[nb[0]][nb[1]].reward
                values.append((nb_reward + (self.gamma * nb_value)))
            return max(values)
        else:
            return 0.0

    def act(self):
        """
        This function applies a chosen action to the agent and makes the Agent
        'act' in the maze environment.
        :@return: void
        """
        action = self.policy.select_action(self.possible_actions)
        self.state = self.maze.step(self.state, action)
