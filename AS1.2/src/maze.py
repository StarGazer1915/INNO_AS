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
        for row in range(len(self.reward_matrix)):
            for col in range(len(self.reward_matrix[row])):
                if type(self.reward_matrix[row][col]) == list:
                    self.environment[row][col] = Point([row, col],
                                                       self.reward_matrix[row][col][0],
                                                       self.reward_matrix[row][col][1])
                else:
                    self.environment[row][col] = Point([row, col], self.reward_matrix[row][col])

    def step(self, state, action):
        new_state = [state[0] + action[0], state[1] + action[1]]
        try:
            check = self.environment[new_state[0]][new_state[1]]
            if new_state[0] < 0 or new_state[1] < 0:
                return state
            else:
                return new_state
        except IndexError:
            return state

    def print_detailed_result_matrix(self):
        print("\nMaze environment: ([Value, Reward])\n")
        for row in self.environment:
            line = "[ "
            for col in row:
                line += f"[{col.value}, {col.reward}] "
            line += "]"
            print(line)
