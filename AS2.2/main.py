import numpy as np
from src.maze import Maze
from src.agent import Agent
from src.policy import Policy


if __name__ == "__main__":
    maze_matrix = np.array([
            [-1, -1, -1, [40, True]],
            [-1, -1, -10, -10],
            [-1, -1, -1, -1],
            [[10, True], -2, -1, -1]
        ], dtype=object)
    starting_state = [3, 2]

    policy_matrix = np.array([
            ['R', 'R', 'R', 'T'],
            ['RU', 'U', 'U', 'U'],
            ['RU', 'U', 'L', 'L'],
            ['T', 'U', 'U', 'LU']
        ], dtype='U4')

    gamma = 0.5
    alpha = 1.0

    m0 = Maze(maze_matrix, starting_state)
    p0 = Policy("on-policy", policy_matrix)
    a0 = Agent(m0.step, m0.actions, (m0.maze_x_size, m0.maze_y_size), starting_state, p0)

    a0.td_learning(gamma, alpha)
    m0.show_matrices()
    a0.show_agent_matrices()
