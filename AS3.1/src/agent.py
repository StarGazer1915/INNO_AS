import numpy as np
from torch import FloatTensor, from_numpy


class Agent:
    def __init__(self, memory, policy, discount, epsilon_decay, sample_size):
        self.memory = memory
        self.policy = policy
        self.reward = 0.
        self.discount = discount
        self.epsilon_decay = epsilon_decay
        self.sample_size = sample_size

    def train(self, available_actions):
        sample_batch = self.memory.sample(self.sample_size)
        for s in sample_batch:
            # ===== Unpack Transition items ===== #
            action, reward, state, new_state, terminated = s[0], s[1], s[2], s[3], s[4]

            # ===== Calculate q values and determine action_prime ===== #
            q_values = self.policy.nn(from_numpy(np.array(state))).tolist()

            q_prime_values = self.policy.nn(from_numpy(np.array(new_state))).tolist()
            action_prime = self.policy.select_action(available_actions, q_prime_values)

            # ===== Calculate target ===== #
            if terminated:
                a_prime_target = float(reward)
            else:
                a_prime_target = float(reward + self.discount * max(q_prime_values))

            # ===== Apply gradient descent ===== #
            target_values = []
            for i in range(len(q_values)):
                if i == action_prime:
                    target_values.append(a_prime_target)
                else:
                    target_values.append(q_values[i])

            m_input = FloatTensor(q_values).requires_grad_()
            target = FloatTensor(target_values)

            loss = self.policy.loss_fn(target, m_input)

            self.policy.opt.zero_grad()
            loss.backward()
            self.policy.opt.step()

    def decay_epsilon(self):
        self.policy.epsilon *= self.epsilon_decay
