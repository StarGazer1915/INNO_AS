import numpy as np
from src.maze import Maze
from src.agent import Agent
from src.policy import Policy


if __name__ == "__main__":
    reward_matrix = np.array([
            [-1, -1, -1, [40, True]],
            [-1, -1, -10, -10],
            [-1, -1, -1, -1],
            [[10, True], -2, -1, -1]
        ], dtype=object)
    start_position = [3, 2]

    m0 = Maze(reward_matrix, start_position)
    p0 = Policy("random", 1.0)
    a0 = Agent(m0, p0)

    a0.value_iteration()
    m0.show_result_matrix()
    a0.show_directions()

    # for i in range(1):
    #     a0.act()
