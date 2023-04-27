import numpy as np
from point import Point


class Maze:
    def __init__(self, reward_matrix, start_point):
        self.reward_matrix = reward_matrix
        self.maze_x_size = len(reward_matrix[0])
        self.maze_y_size = len(reward_matrix)
        self.start_point = start_point
        self.environment = np.zeros((self.maze_x_size, self.maze_y_size), dtype=object)
        self.actions = {
            "0": [0, -1],  # Left
            "1": [0, +1],  # Right
            "2": [-1, 0],  # Up
            "3": [+1, 0]   # Down
            }
        self.setup_environment()

    def setup_environment(self):
        """
        This function sets the maze environment existing of Point() objects that act
        as possible states for the agent. Each point will have a value, reward value and
        also a boolean value in case the state is a terminal state. The function uses the
        reward matrix to build the environment.
        :@return: void
        """
        for row in range(len(self.reward_matrix)):
            for col in range(len(self.reward_matrix[row])):
                if type(self.reward_matrix[row][col]) == list:
                    self.environment[row][col] = Point([row, col],
                                                       self.reward_matrix[row][col][0],
                                                       self.reward_matrix[row][col][1])
                else:
                    self.environment[row][col] = Point([row, col], self.reward_matrix[row][col])

    def step(self, state, action):
        """
        This function simulates the agent taking a step inside the maze environment
        and getting a new state based on its previous state and its chosen action.
        Should the action result in treading outside the bounds of the maze environment
        then the function will make the agent stay on the same state (coordinate).
        @param state: list [y, x]
        @param action: list [y, x]
        @return: list [y, x]
        """
        new_state = [state[0] + action[0], state[1] + action[1]]
        try:
            check_if_inside_maze = self.environment[new_state[0]][new_state[1]]
            if new_state[0] < 0 or new_state[1] < 0:
                return state
            else:
                return new_state
        except IndexError:
            return state

    def print_detailed_result_matrix(self):
        """
        This function prints a detailed overview of the current
        values and rewards of each point in the maze environment.
        :@return: void
        """
        print("\nMaze environment: ([Value, Reward])\n")
        for row in self.environment:
            line = "[ "
            for col in row:
                line += f"[{col.value}, {col.reward}] "
            line += "]"
            print(line)
