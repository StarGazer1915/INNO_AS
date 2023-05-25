import numpy as np


class Agent:
    def __init__(self, step_func, actions, env_size, start_point, policy_object):
        self.maze_step = step_func
        self.actions = actions
        self.start_position = start_point
        self.state = start_point
        self.value_matrix = np.zeros((env_size[0], env_size[1]), dtype=float)
        self.policy = policy_object

    def tabular_td_zero(self, gamma=1., alpha=1.):
        count_for = 0
        count_while = 0
        for i in range(10):
            previous_matrix = self.value_matrix.copy()
            terminal = False
            while not terminal:
                observation = self.act()    # Observation return: (new_state, terminal, reward)
                new_state = observation[0]
                current_value = self.value_matrix[self.state[0]][self.state[1]]
                next_value = self.value_matrix[new_state[0]][new_state[1]]
                new_value = current_value + alpha * (observation[2] + gamma * next_value - current_value)
                self.value_matrix[self.state[0]][self.state[1]] = new_value

                self.state = new_state
                terminal = observation[1]
                count_while += 1

            count_for += 1
            self.state = self.start_position
            if np.array_equal(previous_matrix, self.value_matrix):
                break

        print(f"\nNo more changes after '{count_for}' episodes and '{count_while}' steps")

    def act(self):
        """
        This function applies a chosen action to the agent and makes the Agent
        'act' in the maze environment. It then returns the information for the
        agent that has been provided by the maze object.
        :@return: void
        """
        action = self.policy.select_action(list(self.actions.keys()), self.state)
        if action == "T":
            return self.state, True, 40
        else:
            return self.maze_step(self.state, self.actions[f"{action}"])

    def show_agent_matrices(self):
        """
        Prints the current generated policy to terminal.
        :@return: void
        """
        print(f"\nAgent value matrix: \n{self.value_matrix}")
        print("\nAgent policy matrix:")
        if len(self.policy.p_matrix) < 1:
            print("'There is no policy available'")
        else:
            print(self.policy.p_matrix)
