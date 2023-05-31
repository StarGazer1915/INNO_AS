import numpy as np
from state import State


class Agent:
    def __init__(self, step_func, actions, env_size, start_point, policy_object):
        self.maze_step = step_func
        self.actions = actions
        self.start_position = start_point
        self.state = start_point
        self.env_size = env_size
        self.state_matrix = {}
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
                current_state_value = self.state_matrix[current_pos].value
                state_prime_value = self.state_matrix[state_prime_pos].value
                new_value = current_state_value + alpha * (result[3] + gamma * state_prime_value - current_state_value)
                self.state_matrix[current_pos].value = new_value

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

    # def sarsa_td_control(self, episodes=1, gamma=1., alpha=1.):
    #     self.state_matrix = []
    #     for y in range(self.env_size[0]):
    #         value_row = []
    #         for x in range(self.env_size[1]):
    #             value_row.append({"L": 0, "R": 0, "U": 0, "D": 0})
    #         self.state_matrix.append(value_row)
    #
    #     count_e = 0
    #     count_s = 0
    #     for i in range(episodes):
    #         available_actions = list(self.actions.keys())
    #         action = self.policy.select_action(available_actions, self.state,
    #                                            self.state_matrix[self.state[0]][self.state[1]])
    #         while True:
    #             observation = self.act(self.state, action)  # Observation return: (old_state, new_state, terminal, reward)
    #             state_prime = observation[0]
    #             self.update_policy(self.state, action)
    #             if not observation[1]:
    #                 action_prime = self.policy.select_action(available_actions, state_prime,
    #                                                          self.state_matrix[state_prime[0]][state_prime[1]])
    #                 current_value = self.state_matrix[self.state[0]][self.state[1]][action]
    #                 value_prime = self.state_matrix[state_prime[0]][state_prime[1]][action_prime]
    #
    #                 # print(f"observation: {observation}")
    #                 # print(f"state_prime: {state_prime}, action: {action}, action_prime: {action_prime}")
    #                 # print(f"current_value: {current_value}, value_prime: {value_prime}")
    #
    #                 new_value = current_value + alpha * (observation[2] + gamma * value_prime - current_value)
    #                 # print(f"\nnew_value = current_value + alpha * (observation[2] + gamma * value_prime - current_value)")
    #                 # print(f"{new_value} = {current_value} + {alpha} * ({observation[2]} + {gamma} * {value_prime} - {current_value})")
    #
    #                 self.state_matrix[self.state[0]][self.state[1]][action] = new_value
    #                 self.state = state_prime
    #                 action = action_prime
    #             else:
    #                 current_value = self.state_matrix[self.state[0]][self.state[1]][action]
    #                 new_value = current_value + alpha * (observation[2] + gamma * 0 - current_value)
    #                 self.state_matrix[self.state[0]][self.state[1]][action] = new_value
    #                 self.update_policy(state_prime, "T")
    #                 self.state = state_prime
    #                 break
    #
    #             count_s += 1
    #         count_e += 1
    #         self.state = self.start_position
    #
    #     print(f"\nDone after '{count_e}' episodes and '{count_s}' steps")

    def check_state_matrix(self, action_result):
        current_pos = action_result[0].state_coordinate
        if str(current_pos) not in self.state_matrix:
            self.state_matrix[str(current_pos)] = action_result[0]

        state_prime_pos = action_result[1].state_coordinate
        if str(state_prime_pos) not in self.state_matrix:
            self.state_matrix[str(state_prime_pos)] = action_result[1]

    def update_policy(self, action_result):
        current_pos = action_result[0].state_coordinate
        self.policy.p_matrix[current_pos[0]][current_pos[1]] = action_result

    def show_agent_matrices(self):
        """
        Prints the current generated policy to terminal.
        :@return: void
        """
        print(f"\nAgent value matrix: ")
        if self.state_matrix[str(self.start_position)].q_table is not None:
            for s in self.state_matrix:
                pos = s.state_coordinate
                for k in s.q_table:
                    s.q_table[k] = round(s.q_table[k], 2)
                    self.state_matrix[str(pos)].q_table = s.q_table
            for state in self.state_matrix:
                line = ""
                for v in list(state.q_table.values()):
                    line += "{0:6}, ".format(v)
                line += "| "
                print(line)
        else:
            for i in range(self.env_size[0]):
                line = ""
                for j in range(self.env_size[1]):
                    line += "{0:6}, ".format(self.state_matrix[f"[{i}, {j}]"].value)
                line += "| "
                print(line)

        # print("\nAgent policy matrix:")
        # if len(self.policy.p_matrix) < 1:
        #     print("'There is no policy available'")
        # else:
        #     print(self.policy.p_matrix)
