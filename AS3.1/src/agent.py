from torch import FloatTensor, from_numpy


class Agent:
    def __init__(self, env_step, memory, policy, device, sample_size, num_epochs, max_steps, discount):
        self.step_function = env_step
        self.memory = memory
        self.policy = policy
        self.device = device
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.discount = discount

    def train(self, state):
        self.policy.nn.train(mode=True)

        # ===== Calculate q-values ===== #
        current_state = from_numpy(state).to(self.device)
        state_q = self.policy.nn(current_state).tolist()

        # ===== Decide action ===== #
        action = self.policy.select_action(state_q)

        # ===== Take action, observe result ===== #
        new_state, reward, terminated, truncated, info = self.step_function(action)

        # ===== Store Transition ===== #
        transition = (state, new_state, action, reward, terminated)
        self.memory.store(transition)

        loss_total = 0
        sample_batch = self.memory.sample(self.sample_size)
        for s in sample_batch:
            # ===== Unpack Transition items ===== #
            s_state, s_next_state = s[0], s[1]
            s_action, s_reward, s_terminated = s[2], s[3], s[4]

            # ===== Calculate current q-values ===== #
            s_state_q = self.policy.nn(from_numpy(s_state).to(self.device)).tolist()
            s_next_state_q = self.policy.nn(from_numpy(s_next_state).to(self.device)).tolist()

            # ===== Calculate target ===== #
            action_prime = self.policy.select_action(s_next_state_q)
            if s_terminated:
                a_prime_target = s_reward
            else:
                a_prime_target = s_reward + self.discount * max(s_next_state_q)

            target_values = []
            for i in range(len(s_state_q)):
                if i == action_prime:
                    target_values.append(a_prime_target)
                else:
                    target_values.append(s_state_q[i])

            # ===== Apply gradient descent ===== #
            m_input = FloatTensor(s_next_state_q).requires_grad_()
            target = FloatTensor(target_values)

            loss = self.policy.loss_fn(m_input, target)
            loss_total += loss.item()
            self.policy.opt.zero_grad()
            loss.backward()
            self.policy.opt.step()

        return new_state, reward, loss_total, terminated, truncated
