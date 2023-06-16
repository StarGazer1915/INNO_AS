import torch
import random
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Agent:
    def __init__(self, memory, policy, device, target_net, sample_size, num_epochs, max_steps, learning_rate, gamma, tau):
        self.memory = memory
        self.policy = policy
        self.device = device
        self.target_net = target_net
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau

    def train(self, batch_size, gamma, tau, weight_modify_chance, lr):

        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy.neural_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.to(torch.float32)

        # Compute Huber loss
        criterion = torch.nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        torch.optim.Adam(self.policy.neural_net.parameters(), lr=lr).zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.neural_net.parameters(), 100)
        torch.optim.Adam(self.policy.neural_net.parameters(), lr=lr).step()

        # DDQN weight overwriting
        if random.random() > weight_modify_chance:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy.neural_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
