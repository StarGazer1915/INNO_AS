import numpy as np
from state import State


class Agent:
    def __init__(self, step_func, start_func, actions, env_size, start_point, policy_object):
        self.maze_step = step_func
        self.maze_get_start_state = start_func
        self.actions = actions
        self.start_position = start_point
        self.state = start_point
        self.env_size = env_size
        self.state_dict = {}
        self.policy = policy_object

    def tabular_td_zero(self, episodes, gamma=1., alpha=1.):
        """
        Function that applies the TD(0) evaluation algorithm.
        @param gamma: float
        @param alpha: float
        @return: void
        """
        count_e = 0
        count_s = 0
        for i in range(episodes):
            deltas = []
            terminal = False
            while not terminal:
                action = self.policy.select_action(list(self.actions.keys()), self.state)
                result = self.maze_step(self.state, action[0])
                self.check_state_matrix(result)

                current_pos, state_prime_pos = str(result[0].state_coordinate), str(result[1].state_coordinate)
                current_state_value = self.state_dict[current_pos].value
                state_prime_value = self.state_dict[state_prime_pos].value
                new_value = current_state_value + alpha * (result[3] + gamma * state_prime_value - current_state_value)
                self.state_dict[current_pos].value = new_value

                delta = current_state_value - new_value
                if delta < 0:
                    delta = delta * -1
                deltas.append(delta)

                self.state = result[1].state_coordinate  # State_prime coordinate
                terminal = result[2]  # State_prime terminal value
                count_s += 1

            count_e += 1
            self.state = self.start_position
            if max(deltas) == 0:
                break

        print(f"\nIt took '{count_e}' episodes and '{count_s}' steps to converge.")

    def sarsa_td_control(self, episodes=1, gamma=1., alpha=1.):
        """
        Function that applies the SARSA algorithm.
        @param episodes: float
        @param gamma: float
        @param alpha: float
        @return: void
        """
        count_e = 0
        count_s = 0
        available_actions = list(self.actions.keys())
        for i in range(episodes):
            terminal = False
            current_state = self.maze_get_start_state()  # State at self.start_position
            action = self.policy.select_action(available_actions, self.state, current_state.q_table)
            while not terminal:
                state_prime = self.maze_step(self.state, action)
                state_prime_pos = str(state_prime.state_coordinate)
                self.add_to_dict_if_needed(str(self.state), str(state_prime.state_coordinate))
                action_prime = self.policy.select_action(available_actions, self.state, state_prime.q_table)

                current_q = self.state_dict[str(self.state)]
                qsa = current_q[action]
                prime_q = self.state_dict[state_prime_pos]
                qsa_prime = prime_q[action_prime]
                # print(f"\ncurrent_q: {current_q}, action: {action}, qsa: {qsa}")
                # print(f"prime_q: {prime_q}, action_prime: {action_prime}, qsa_prime: {qsa_prime}")

                # print(f"\n{self.state_dict}")
                self.state_dict[str(self.state)][action] = qsa + alpha * (state_prime.reward + gamma * qsa_prime - qsa)
                # print(self.state_dict)

                terminal = state_prime.terminal
                action = action_prime.copy()
                self.state = state_prime.state_coordinate

            count_e += 1
            self.state = self.start_position

        print(f"\nDone after '{count_e}' episodes and '{count_s}' steps")

    def add_to_dict_if_needed(self, current_state_pos, state_prime_pos):
        if current_state_pos not in self.state_dict:
            self.state_dict[current_state_pos] = {"L": 0, "R": 0, "U": 0, "D": 0}
        if state_prime_pos not in self.state_dict:
            self.state_dict[state_prime_pos] = {"L": 0, "R": 0, "U": 0, "D": 0}

    def update_policy(self, action_result, chosen_action):
        current_pos = action_result[0].state_coordinate
        state_prime_pos = action_result[1].state_coordinate
        if action_result[1].terminal:
            self.policy.p_matrix[state_prime_pos[0]][state_prime_pos[1]] = "T"
        self.policy.p_matrix[current_pos[0]][current_pos[1]] = chosen_action

    def show_agent_matrices(self):
        """
        Prints the current generated policy to terminal.
        :@return: void
        """
        print(f"\nAgent value matrix: ")
        if len(self.state_dict) != 0:
            for i in range(self.env_size[0]):
                line = "|"
                for j in range(self.env_size[1]):
                    try:
                        values = list(self.state_dict[f"[{i}, {j}]"].values())
                        line += "{0:7} {1:7} {2:7} {3:7}   | ".format(round(values[0], 2), round(values[1]),
                                                                      round(values[2], 2), round(values[3]))
                    except:
                        line += "{0:7} {1:7} {2:7} {3:7}   | ".format(0, 0, 0, 0)
                print(line)
        else:
            for i in range(self.env_size[0]):
                line = ""
                for j in range(self.env_size[1]):
                    line += "{0:6}, ".format(round(self.state_dict[f"[{i}, {j}]"].value, 2))
                line += "| "
                print(line)

        print("\nAgent policy matrix:")
        if len(self.policy.p_matrix) < 1:
            print("'There is no policy available'")
        else:
            print(self.policy.p_matrix)
