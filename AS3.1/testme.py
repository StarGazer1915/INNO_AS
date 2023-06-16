import numpy as np
import gymnasium as gym
from random import randint
from collections import namedtuple
import torch

from src.dqn import DQN
from src.agent import Agent
from src.policy import Policy
from src.memory import Memory

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

num_epochs = 1000
max_steps = 2000
avg_reward_threshold = 200

learning_rate = 0.001
epsilon_start = 0.9
epsilon_end = 0.01
epsilon_decay = 0.99
gamma = 0.99
tau = 0.005

memory_size = 20000
sample_size = 32

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env = gym.make("LunarLander-v2", render_mode=None)

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

print(policy_net)
print(target_net)

# Memory class for the Agent
me0 = Memory(memory_size)

# The Policy class for the Agent
p0 = Policy(policy_net, device, env.action_space, epsilon_start, epsilon_end, epsilon_decay)

# The Agent class
a0 = Agent(me0, p0, device, target_net, sample_size, num_epochs, max_steps, learning_rate, gamma, tau)

a0.policy.neural_net.train(mode=False)
a0.target_net.train(mode=False)

for i in range(10):  # 10 epochs to fill memory
    state = torch.tensor(env.reset(seed=randint(0, 1000))[0], dtype=torch.float32, device=device).unsqueeze(0)
    for step in range(max_steps):
        # ===== Decide action ===== #
        action = a0.policy.select_action(state)

        # ===== Take action, observe result ===== #
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # ===== Store Transition ===== #
        a0.memory.store(state, action, next_state, reward)
        state = next_state

        if terminated or truncated:
            break

env.close()

print(len(a0.memory.deque))

env = gym.make("LunarLander-v2", render_mode=None)

a0.policy.neural_net.train(mode=True)
a0.target_net.train(mode=True)

rewards = []
for i in range(num_epochs + 1):
    epoch_reward = 0
    state = torch.tensor(env.reset(seed=randint(0, 1000))[0], dtype=torch.float32, device=device).unsqueeze(0)
    for step in range(max_steps):
        # ===== Decide action ===== #
        action = a0.policy.select_action(state)

        # ===== Take action, observe result ===== #
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        epoch_reward += reward

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # ===== Store Transition ===== #
        a0.memory.store(state, action, next_state, reward)
        state = next_state

        # ===== Train the model ===== #
        a0.train(sample_size, 0.9, 0.0001, 0.5, 0.001)

        if terminated or truncated:
            break

    rewards.append(epoch_reward)

    # ===== Visualization ===== #
    print(f"Epoch {i} | Epoch rewards: {epoch_reward} | Epsilon: {a0.policy.epsilon}")

    if i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        run_avg_reward = np.mean(rewards)
        if run_avg_reward >= 200:
            print(
                f"\nTraining done at Epoch {i} | Average reward: {run_avg_reward} | Epsilon is now: {a0.policy.epsilon}\n")
            rewards = []
            break
        else:
            print(f"\nEpoch {i - 100}-{i} | Average reward: {run_avg_reward} | Epsilon is now: {a0.policy.epsilon}\n")
            rewards = []

env.close()
