import numpy as np
import gymnasium as gym
from random import randint
import torch

from src.lmodel import Lmodel
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
epsilon = 1.0
epsilon_decay = 0.99
discount = 0.99
memory_size = 32000
sample_size = 64
available_actions = [0,1,2,3]

env = gym.make("LunarLander-v2", render_mode=None)

# Memory class for the Agent
me0 = Memory(memory_size)

# The Policy class for the Agent
p0 = Policy(Lmodel().to(device), learning_rate, epsilon, available_actions, epsilon_decay)

# The Agent class
a0 = Agent(env.step, me0, p0, device, sample_size, num_epochs, max_steps, discount)

rewards = []
losses = []

state, info = env.reset(seed=randint(0, 1000))
for i in range(num_epochs):
    epoch_reward = 0
    epoch_loss = 0
    for step in range(max_steps):
        state, reward, loss, terminated, truncated = a0.train(state)
        epoch_reward += reward
        epoch_loss += loss

        if terminated or truncated:
            state, info = env.reset(seed=randint(0, 1000))
            break

    a0.policy.decay()
    rewards.append(epoch_reward)
    losses.append(epoch_loss)

    # ===== Visualization ===== #
    print(f"Epoch {i} | Epoch rewards: {epoch_reward} | Training losses: {epoch_loss} | Epsilon: {a0.policy.epsilon}")

    if i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        run_avg_reward = np.mean(rewards)
        run_avg_loss = np.mean(losses)
        if run_avg_reward >= 200:
            print(
                f"Epoch {i} | Average reward: {run_avg_reward} and Loss: {run_avg_loss} | Epsilon: {a0.policy.epsilon}\n")
            break
        else:
            print(
                f"Epoch {i} | Average reward: {run_avg_reward} and Loss: {run_avg_loss} | Epsilon: {a0.policy.epsilon}\n")

env.close()
