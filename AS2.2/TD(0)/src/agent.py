

class Agent:
    def __init__(self, step_func, actions, env_size, start_point, policy_object):
        self.maze_step = step_func
        self.actions = actions
        self.start_position = start_point
        self.state = start_point
        self.env_size = env_size
        self.state_dict = {}
        self.policy = policy_object

    def tabular_td_zero(self, episodes=1, gamma=1., alpha=1.):
        """
        Function that applies the TD(0) evaluation algorithm.
        @param episodes: int
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
                action = self.policy.select_action(self.state)
                result = self.maze_step(self.state, action)  # prime_coordinate, prime_reward, prime_terminal
                current_pos, state_prime_pos = str(self.state), str(result[0])
                self.add_to_dict_if_needed(current_pos, state_prime_pos)

                current_value = self.state_dict[current_pos]
                state_prime_value = self.state_dict[state_prime_pos]
                new_value = current_value + alpha * (result[1] + gamma * state_prime_value - current_value)
                self.state_dict[current_pos] = new_value

                delta = current_value - new_value
                if delta < 0:
                    delta = delta * -1
                deltas.append(delta)

                self.state = result[0]
                terminal = result[2]
                count_s += 1

            count_e += 1
            self.state = self.start_position
            if max(deltas) == 0:
                break

        print(f"\nNo more changes after '{count_e}' episodes and '{count_s}' steps.")

    def add_to_dict_if_needed(self, current_state_pos, state_prime_pos):
        """
        Functions that checks if newly discovered states are in the
        state_dict. If not, then it adds them for use in calculation.
        @param current_state_pos: str
        @param state_prime_pos: str
        @return: void
        """
        if current_state_pos not in self.state_dict:
            self.state_dict[current_state_pos] = 0
        if state_prime_pos not in self.state_dict:
            self.state_dict[state_prime_pos] = 0

    def show_agent_matrices(self):
        """
        Prints the current generated agent state_dict
        and policy to terminal for visualization purposes.
        :@return: void
        """
        print(f"\nAgent value matrix: ")
        for i in range(self.env_size[0]):
            line = "|"
            for j in range(self.env_size[1]):
                try:
                    line += "{0:6}, ".format(round(self.state_dict[f"[{i}, {j}]"], 2))
                except:
                    line += "{0:6}, ".format(0)
            line += "| "
            print(line)

        print("\nAgent policy matrix:")
        if len(self.policy.p_matrix) < 1:
            print("'There is no policy available'")
        else:
            print(self.policy.p_matrix)
