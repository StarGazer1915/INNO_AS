import numpy as np


class Agent:
    def __init__(self, maze_object, policy_object):
        self.maze = maze_object
        self.state = self.maze.start_point
        self.policy = policy_object
        self.action_values = list(self.maze.actions.values())
        self.action_keys = list(self.maze.actions.keys())
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
        for i in range(len(self.action_values)):
            nb = [state_coordinate[0] + self.action_values[i][0], state_coordinate[1] + self.action_values[i][1]]
            try:
                check_if_inside_maze = self.maze.environment[nb[0]][nb[1]]
                if nb[0] >= 0 and nb[1] >= 0:
                    neighbouring_state_coordinates.append([nb, self.action_keys[i]])
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

        print(f"\nValue Iteration took '{iter_count}' total iterations (including iteration 0).")

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
                nb_value = self.maze.environment[nb[0][0]][nb[0][1]].value
                nb_reward = self.maze.environment[nb[0][0]][nb[0][1]].reward
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
        action = self.policy.select_action(self.action_values)
        self.state = self.maze.step(self.state, action)

    def show_directions(self):
        """
        This function prints a detailed overview of the directions the agent would go in
        the maze environment should the agent not be randomly performing actions and follow
        the optimal policy by choosing the best route in each state.
        :@return: void
        """
        print("\nDirections: ")
        for row in range(len(self.maze.environment)):
            line = []
            for col in range(len(self.maze.environment[row])):
                if self.maze.environment[row][col].terminal:
                    direction = "X"
                else:
                    neighbours = self.get_reachable_states([row, col])
                    routes = {}
                    for nb in neighbours:
                        own_value = self.maze.environment[row][col].value
                        nb_value = self.maze.environment[nb[0][0]][nb[0][1]].value
                        nb_reward = self.maze.environment[nb[0][0]][nb[0][1]].reward
                        # If the state is terminal and has a higher or equal reward than the value of
                        # the current state, remove all other routes and go straight for that state.
                        if self.maze.environment[nb[0][0]][nb[0][1]].terminal and nb_reward >= own_value:
                            routes = {nb[1]: nb_reward}
                            break
                        else:
                            # Check if there are multiple states (directions) with the same values
                            # and if there is a new highest value, remove the lower values.
                            if len(routes) < 1:
                                routes[nb[1]] = nb_value
                            else:
                                routes2 = routes.copy()
                                for item in routes2:
                                    if routes[item] == nb_value:
                                        routes[nb[1]] = nb_value
                                    elif routes[item] < nb_value:
                                        routes.pop(item)
                                        routes[nb[1]] = nb_value

                    direction = ""
                    for i in routes:
                        direction += f"{i}"
                line.append(f"{direction}")
            print('| {:2} | {:^2} | {:>2} | {:<2} |'.format(*line))
