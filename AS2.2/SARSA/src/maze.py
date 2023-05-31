import numpy as np
from state import State


class Maze:
    def __init__(self, maze_matrix, start_point):
        self.maze_matrix = maze_matrix
        self.start_point = start_point
        self.maze_y_size = len(maze_matrix)
        self.maze_x_size = len(maze_matrix[0])
        self.environment = np.zeros((self.maze_x_size, self.maze_y_size), dtype=object)
        self.actions = {
            "L": [0, -1],   # Left
            "R": [0, +1],   # Right
            "U": [-1, 0],   # Up
            "D": [+1, 0]    # Down
            }
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
                    self.environment[row][col] = State([row, col],
                                                       self.maze_matrix[row][col][0],  # Reward
                                                       self.maze_matrix[row][col][1])  # Terminal
                else:
                    self.environment[row][col] = State([row, col], self.maze_matrix[row][col])

    def step(self, coordinate, chosen_action):
        """
        This function simulates the agent taking a step inside the maze environment
        and getting a new state based on its previous state and its chosen action.
        Should the action result in treading outside the bounds of the maze environment
        then the function will make the agent stay on the same state (coordinate).
        @param coordinate: list [y, x]
        @param chosen_action: str
        @return: list [y, x], float, bool
        """
        action = self.actions[chosen_action]
        prime_coordinate = [coordinate[0] + action[0], coordinate[1] + action[1]]
        current_reward = self.environment[coordinate[0]][coordinate[1]].reward
        current_terminal = self.environment[coordinate[0]][coordinate[1]].terminal
        try:
            check_if_inside_maze = self.environment[prime_coordinate[0]][prime_coordinate[1]]
            if prime_coordinate[0] < 0 or prime_coordinate[1] < 0:
                return [coordinate, current_reward, current_terminal]
            else:
                prime_reward = self.environment[prime_coordinate[0]][prime_coordinate[1]].reward
                prime_terminal = self.environment[prime_coordinate[0]][prime_coordinate[1]].terminal
                return [prime_coordinate, prime_reward, prime_terminal]
        except IndexError:
            return [coordinate, current_reward, current_terminal]
