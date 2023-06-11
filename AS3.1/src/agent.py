import numpy as np
from torch import from_numpy


class Agent:
    def __init__(self, memory, policy, decay, sample_size):
        self.memory = memory
        self.policy = policy
        self.reward = 0.
        self.decay = decay
        self.sample_size = sample_size

    def train(self, available_actions):
        sample_batch = self.memory.sample(self.sample_size)
        for s in sample_batch:
            # ===== Unpack Transition items ===== #
            action, reward, state, new_state, terminated = s[0], s[1], s[2], s[3], s[4]
            if terminated:
                reward = 0.

            # ===== Calculate q values and determine action_prime ===== #
            q_values = self.policy.nn(from_numpy(np.array(state))).tolist()
            q_prime_values = self.policy.nn(from_numpy(np.array(new_state))).tolist()
            action_prime = self.policy.select_action(available_actions, q_prime_values)
            a_prime_target = reward + self.decay * q_prime_values[action_prime]

            # print(f"\nq_values: \n{q_values}")
            # print(f"\nq_prime_values: \n{q_prime_values}")
            # print(f"\naction_prime: {action_prime}")
            # print(f"\n{a_prime_target} = {reward} + {self.decay} * {q_prime_values[action_prime]}")
