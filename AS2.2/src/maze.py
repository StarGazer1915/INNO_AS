import numpy as np


class Maze:
    def __init__(self, maze_matrix, start_point):
        self.maze_matrix = maze_matrix
        self.start_point = start_point
        self.maze_y_size = len(maze_matrix)
        self.maze_x_size = len(maze_matrix[0])
        self.reward_matrix = np.zeros((self.maze_x_size, self.maze_y_size), dtype=float)
        self.terminal_matrix = np.zeros((self.maze_x_size, self.maze_y_size), dtype=bool)
        self.actions = {
            "L": [0, -1],   # Left
            "R": [0, +1],   # Right
            "U": [-1, 0],   # Up
            "D": [+1, 0]    # Down
            }
        self.action_values = list(self.actions.values())
        self.action_keys = list(self.actions.keys())
        self.setup_environment()

    def setup_environment(self):
        """
        This function builds the maze environment by building three matrices (value, reward, terminal)
        from the given maze_matrix attribute.
        :@return: void
        """
        for row in range(len(self.maze_matrix)):
            for col in range(len(self.maze_matrix[row])):
                if type(self.maze_matrix[row][col]) == list:
                    self.reward_matrix[row][col] = float(self.maze_matrix[row][col][0])
                    self.terminal_matrix[row][col] = bool(self.maze_matrix[row][col][1])
                else:
                    self.reward_matrix[row][col] = float(self.maze_matrix[row][col])

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
                check_if_inside_maze = self.reward_matrix[nb[0]][nb[1]]
                if nb[0] >= 0 and nb[1] >= 0:
                    neighbouring_state_coordinates.append([nb, self.action_keys[i]])
            except IndexError:
                pass

        return neighbouring_state_coordinates

    def step(self, state, chosen_action):
        """
        This function simulates the agent taking a step inside the maze environment
        and getting a new state based on its previous state and its chosen action.
        Should the action result in treading outside the bounds of the maze environment
        then the function will make the agent stay on the same state (coordinate).
        @param state: list [y, x]
        @param chosen_action: str
        @return: (list [y, x], bool, float, str)
        """
        state_terminal = self.terminal_matrix[state[0]][state[1]]
        state_reward = self.reward_matrix[state[0]][state[1]]
        if chosen_action != "T":
            action = self.actions[chosen_action]
            new_state = [state[0] + action[0], state[1] + action[1]]
            try:
                check_if_inside_maze = self.reward_matrix[new_state[0]][new_state[1]]
                if new_state[0] < 0 or new_state[1] < 0:
                    return state, state_terminal, state_reward, action
                else:
                    return new_state, self.terminal_matrix[new_state[0]][new_state[1]], \
                           self.reward_matrix[new_state[0]][new_state[1]], action
            except IndexError:
                return state, state_terminal, state_reward, action
        else:
            return state, state_terminal, state_reward, [0, 0]

    def show_matrices(self):
        names = ["Terminal matrix:", "Reward matrix:"]
        matrices = [self.terminal_matrix, self.reward_matrix]
        for i in range(len(matrices)):
            print(f"\n{names[i]}\n{matrices[i]}")
