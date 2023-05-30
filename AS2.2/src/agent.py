import numpy as np


class Agent:
    def __init__(self, step_func, actions, env_size, start_point, policy_object):
        self.maze_step = step_func
        self.actions = actions
        self.start_position = start_point
        self.state = start_point
        self.env_size = env_size
        self.value_matrix = None
        self.reward_matrix = None
        self.policy = policy_object

    def tabular_td_zero(self, gamma=1., alpha=1.):
        """
        Function that applies the TD(0) evaluation algorithm.
        @param gamma: float
        @param alpha: float
        @return: void
        """
        self.value_matrix = np.zeros((self.env_size[0], self.env_size[1]), dtype=float)
        count_e = 0
        count_s = 0
        while True:
            previous_matrix = self.value_matrix.copy()
            terminal = False
            while not terminal:
                action = self.policy.select_action(list(self.actions.keys()), self.state)
                observation = self.act(self.state, action)  # Observation return: (new_state, terminal, reward)
                new_state = observation[0]
                current_value = self.value_matrix[self.state[0]][self.state[1]]
                next_value = self.value_matrix[new_state[0]][new_state[1]]
                new_value = current_value + alpha * (observation[2] + gamma * next_value - current_value)
                self.value_matrix[self.state[0]][self.state[1]] = new_value

                self.state = new_state
                terminal = observation[1]
                count_s += 1

            count_e += 1
            self.state = self.start_position
            if np.array_equal(previous_matrix, self.value_matrix):
                break

        print(f"\nNo more changes after '{count_e}' episodes and '{count_s}' steps")

    def sarsa_td_control(self, episodes=1, gamma=1., alpha=1.):
        self.reward_matrix = []
        self.value_matrix = []
        for y in range(self.env_size[0]):
            reward_row = []
            value_row = []
            for x in range(self.env_size[1]):
                # np.random.choice(40)
                value_row.append({"L": np.random.choice(40), "R": np.random.choice(40),
                                  "U": np.random.choice(40), "D": np.random.choice(40)})
                reward_row.append(0)
            self.reward_matrix.append(reward_row)
            self.value_matrix.append(value_row)

        count_e = 0
        count_s = 0
        for i in range(episodes):
            available_actions = list(self.actions.keys())
            action = self.policy.select_action(available_actions, self.state,
                                               self.value_matrix[self.state[0]][self.state[1]])
            while True:
                observation = self.act(self.state, action)  # Observation return: (new_state, terminal, reward)
                state_prime = observation[0]
                action_prime = self.policy.select_action(available_actions, state_prime,
                                                         self.value_matrix[state_prime[0]][state_prime[1]])
                current_value = self.value_matrix[self.state[0]][self.state[1]][action]
                value_prime = self.value_matrix[state_prime[0]][state_prime[1]][action_prime]

                print(f"observation: {observation}")
                print(f"state_prime: {state_prime}, action: {action}, action_prime: {action_prime}")
                print(f"current_value: {current_value}, value_prime: {value_prime}")

                new_value = current_value + alpha * (observation[2] + gamma * value_prime - current_value)
                print(f"\nnew_value = current_value + alpha * (observation[2] + gamma * value_prime - current_value)")
                print(f"{new_value} = {current_value} + {alpha} * ({observation[2]} + {gamma} * {value_prime} - {current_value})")

                ch1 = 0



                break







                # self.reward_matrix[state_prime[0]][state_prime[1]] = observation[2]
                # action_prime = self.policy.select_action(available_actions, state_prime)
                #
                # if action_prime != "T":
                #     current_value = self.value_matrix[self.state[0]][self.state[1]][action]  # Q(S, A)
                #     next_value = self.value_matrix[state_prime[0]][state_prime[1]][action_prime]  # Q(S', A')
                #
                #     new_value = current_value + alpha * (observation[2] + gamma * next_value - current_value)
                #     self.value_matrix[self.state[0]][self.state[1]][action] = new_value
                #
                #     directions = self.value_matrix[self.state[0]][self.state[1]]
                #     self.policy.p_matrix[self.state[0]][self.state[1]] = max(directions, key=directions.get)
                #     self.state = state_prime
                #     action = action_prime
                # else:
                #     # self.value_matrix[self.state[0]][self.state[1]][action] = observation[2]
                #     self.value_matrix[state_prime[0]][state_prime[1]] = {"L": 0, "R": 0, "U": 0, "D": 0}
                #     self.policy.p_matrix[state_prime[0]][state_prime[1]] = "T"
                #     self.state = state_prime

                if observation[1]:
                    self.policy.p_matrix[state_prime[0]][state_prime[1]] = "T"
                    break

                count_s += 1

            count_e += 1
            self.state = self.start_position

        print(f"\nDone after '{count_e}' episodes and '{count_s}' steps")

    def q_learning(self, episodes=1, gamma=1., alpha=1.):
        self.reward_matrix = []
        self.value_matrix = []
        for y in range(self.env_size[0]):
            reward_row = []
            value_row = []
            for x in range(self.env_size[1]):
                # np.random.choice(40)
                value_row.append({"L": 0, "R": 0, "U": 0, "D": 0})
                reward_row.append(0)
            self.reward_matrix.append(reward_row)
            self.value_matrix.append(value_row)

        count_e = 0
        count_s = 0
        action = None
        for i in range(episodes):
            terminal = False
            while not terminal:
                if not action:
                    action = self.policy.select_action(list(self.actions.keys()), self.state)
                observation = self.act(self.state, action)  # Observation return: (new_state, terminal, reward)
                state_prime = observation[0]
                if observation[1]:
                    self.policy.p_matrix[state_prime[0]][state_prime[1]] = "T"
                self.reward_matrix[state_prime[0]][state_prime[1]] = observation[2]
                action_prime = self.policy.select_action(list(self.actions.keys()), state_prime)

                if action_prime != "T":
                    current_value = self.value_matrix[self.state[0]][self.state[1]][action]
                    next_value = self.value_matrix[state_prime[0]][state_prime[1]][action_prime]
                    new_value = current_value + alpha * (observation[2] + gamma * next_value - current_value)
                    self.value_matrix[self.state[0]][self.state[1]][action] = new_value

                    directions = self.value_matrix[self.state[0]][self.state[1]]
                    self.policy.p_matrix[self.state[0]][self.state[1]] = max(directions, key=directions.get)
                    self.state = state_prime
                    action = action_prime
                    terminal = observation[1]
                else:
                    self.value_matrix[state_prime[0]][state_prime[1]] = {"L": 0, "R": 0, "U": 0, "D": 0}
                    self.policy.p_matrix[state_prime[0]][state_prime[1]] = "T"
                    self.state = state_prime
                    terminal = observation[1]
                count_s += 1

            count_e += 1
            self.state = self.start_position

        print(f"\nDone after '{count_e}' episodes and '{count_s}' steps")

    def act(self, current_state, chosen_action):
        """
        This function applies a chosen action to the agent and makes the Agent
        'act' in the maze environment. It then returns the information for the
        agent that has been provided by the maze object.
        :@return: void
        """
        return self.maze_step(current_state, chosen_action)

    def show_agent_matrices(self):
        """
        Prints the current generated policy to terminal.
        :@return: void
        """
        print(f"\nAgent reward matrix: ")
        for row in self.reward_matrix:
            print(row)
        print(f"\nAgent value matrix: ")
        if type(self.value_matrix[0][0]) == dict:
            for r in self.value_matrix:
                for d in r:
                    for k in d:
                        d[k] = round(d[k], 2)
            for row in self.value_matrix:
                line = ""
                for d in row:
                    values = list(d.values())
                    line += "{0:6}, {1:6}, {2:6}, {3:6} | ".format(values[0], values[1], values[2], values[3])
                print(line)
        else:
            print(self.value_matrix)
        print("\nAgent policy matrix:")
        if len(self.policy.p_matrix) < 1:
            print("'There is no policy available'")
        else:
            print(self.policy.p_matrix)
