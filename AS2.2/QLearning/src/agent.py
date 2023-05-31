

class Agent:
    def __init__(self, step_func, actions, env_size, start_point, policy_object):
        self.maze_step = step_func
        self.actions = actions
        self.start_position = start_point
        self.state = start_point
        self.env_size = env_size
        self.state_dict = {}
        self.policy = policy_object

    def sarsa_td_control(self, episodes=1, gamma=1., alpha=1.):
        """
        Function that applies the SARSA (on-policy TD control) algorithm.
        @param episodes: float
        @param gamma: float
        @param alpha: float
        @return: void
        """
        count_e = 0
        count_s = 0
        available_actions = list(self.actions.keys())
        self.add_to_dict_if_needed(self.start_position)
        start_state_q = self.state_dict[str(self.state)]
        for i in range(episodes):
            terminal = False
            action = self.policy.select_action(available_actions, start_state_q)
            while not terminal:
                result = self.maze_step(self.state, action)  # prime_coordinate, prime_reward, prime_terminal
                self.add_to_dict_if_needed([self.state, result[0]])
                prime_coordinate = str(result[0])
                action_prime = self.policy.select_action(available_actions, self.state_dict[prime_coordinate])
                self.update_policy(self.state, result[0], result[2], action)

                qsa = self.state_dict[str(self.state)][action]
                prime_qsa = self.state_dict[prime_coordinate][action_prime]
                self.state_dict[str(self.state)][action] = qsa + alpha * (result[1] + gamma * prime_qsa - qsa)

                self.state = result[0]
                action = action_prime
                terminal = result[2]
                count_s += 1

            count_e += 1
            self.state = self.start_position

        print(f"\nDone after '{count_e}' episodes and '{count_s}' steps")

    def add_to_dict_if_needed(self, state_coordinates):
        """
        Functions that checks if newly discovered states are in the
        state_dict. If not, then it adds them for use in calculation.
        @param state_coordinates: nested list or list [y, x]
        @return: void
        """
        if type(state_coordinates) == list and type(state_coordinates[0]) == list:
            for c in state_coordinates:
                if str(c) not in self.state_dict:
                    self.state_dict[str(c)] = {"L": 0, "R": 0, "U": 0, "D": 0}
        else:
            if str(state_coordinates) not in self.state_dict:
                self.state_dict[str(state_coordinates)] = {"L": 0, "R": 0, "U": 0, "D": 0}

    def update_policy(self, current_pos, state_prime_pos, state_prime_terminal, chosen_action):
        """
        Updates the policy matrix with the chosen action of the agent.
        @param current_pos: list [y, x]
        @param state_prime_pos: list [y, x]
        @param state_prime_terminal: bool
        @param chosen_action: str
        @return: void
        """
        if state_prime_terminal:
            self.policy.p_matrix[state_prime_pos[0]][state_prime_pos[1]] = "T"
        self.policy.p_matrix[current_pos[0]][current_pos[1]] = chosen_action

    def show_agent_matrices(self):
        """
        Visualization function for the command line.
        @return: void
        """
        print(f"\nAgent Q function of state_dict: ")
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
