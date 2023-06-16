import torch
from torch import tensor, DoubleTensor


class Agent:
    def __init__(self, memory, policy, device, sample_size, num_epochs, max_steps, discount):
        self.memory = memory
        self.policy = policy
        self.device = device
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.discount = discount

    def train(self):
        sample_batch = self.memory.sample(self.sample_size)
        for s in sample_batch:
            # ===== Unpack Transition items ===== #
            s_state, s_next_state = s[0], s[1]
            s_action, s_reward, s_terminated = s[2], s[3], s[4]

            # ===== Calculate current q-values ===== #
            s_next_state_q = self.policy.nn(s_next_state).detach().numpy().copy()

            # ===== Define best action in next state ===== #
            a_index = 0
            highest = max(s_next_state_q)
            for i in range(len(s_next_state_q)):
                if s_next_state_q[i] >= highest:
                    a_index = i
                    highest = s_next_state_q[i]

            # ===== Calculate target ===== #
            if s_terminated:
                a_prime_target = s_reward
            else:
                a_prime_target = s_reward + self.discount * s_next_state_q[a_index]

            # ===== Apply gradient descent ===== #
            minput = tensor([s_next_state_q[a_index]], requires_grad=True).to(self.device, torch.float32)
            target = tensor([a_prime_target]).to(torch.float32).to(self.device, torch.float32)

            loss = self.policy.loss_fn(minput, target)
            self.policy.opt.zero_grad()
            loss.backward()
            self.policy.opt.step()
