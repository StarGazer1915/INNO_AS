import numpy as np
import time
from src.maze import Maze
from src.agent import Agent
from src.policy import Policy


if __name__ == "__main__":
    start_time = time.perf_counter()
    maze_matrix = np.array([
            [-1, -1, -1, [40, True]],
            [-1, -1, -10, -10],
            [-1, -1, -1, -1],
            [[10, True], -2, -1, -1]
        ], dtype=object)

    starting_state = [3, 2]

    gamma = 1.0
    alpha = 0.3
    epsilon = 0.9

    m0 = Maze(maze_matrix, starting_state)
    maze_size = (m0.maze_x_size, m0.maze_y_size)

    p0 = Policy("SARSA", maze_size, np.array([]), epsilon)
    a0 = Agent(m0.step, m0.actions, maze_size, starting_state, p0)

    a0.sarsa_td_control(250000, gamma, alpha)

    a0.show_agent_matrices()

    print(f"\nTotal execution time (seconds): {time.perf_counter() - start_time:0.4f}")
