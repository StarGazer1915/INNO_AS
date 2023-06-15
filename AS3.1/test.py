import numpy as np
import gymnasium as gym
from random import randint

import torch
from torch import nn, save, load, from_numpy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from src.lmodel import Lmodel

from src.agent import Agent
from src.policy import Policy
from src.memory import Memory
from src.lmodel import Lmodel

num_epochs = 1000
max_steps = 2000
avg_reward_threshold = 200

learning_rate = 0.01
epsilon = 1.0
epsilon_decay = 0.99
discount = 0.99

memory_size = 32000
sample_size = 64

my_nn = Lmodel()
optimizer = Adam(my_nn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

p0 = Policy(my_nn, optimizer, loss_fn, epsilon)
me0 = Memory(memory_size)
a0 = Agent(me0, p0, discount, epsilon_decay, sample_size)

env = gym.make("LunarLander-v2", render_mode=None)
available_actions = [0, 1, 2, 3]

rewards = []
for i in range(num_epochs):
    step_rewards = []
    state, info = env.reset(seed=randint(0, 1000))
    for step in range(max_steps):
        q_values = a0.policy.nn(from_numpy(state)).tolist()

        # ===== Decide action ===== #
        action = a0.policy.select_action(available_actions, q_values)

        # ===== Take action, observe result ===== #
        new_state, reward, terminated, truncated, info = env.step(action)
        step_rewards.append(reward)

        # ===== Store Transition ===== #
        transition = (action, reward, state, new_state, terminated)
        a0.memory.store(transition)

        # ===== Train NN ===== #
        a0.train(available_actions)

        state = new_state

        if terminated or truncated:
            break

    rewards.append(sum(step_rewards))
    a0.decay_epsilon()

    print(f"Epoch {i} | Sum step rewards: {sum(step_rewards)} | Epsilon: {a0.policy.epsilon}")

    if i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        run_avg_reward = np.mean(rewards)
        if run_avg_reward >= 200:
            print(f"\nEpoch {i} | Average Reward: {run_avg_reward} | Epsilon: {a0.policy.epsilon}\n")
            rewards = []
            break
        else:
            print(f"\nEpoch {i} | Average Reward: {run_avg_reward} | Epsilon: {a0.policy.epsilon}\n")
            rewards = []

env.close()

# ===== TRAINING ===== #
# for epoch in range(10):
#     for batch in dataset:
#         X,y = batch
#         X, y = X.to('cuda'), y.to('cuda')
#         yhat = my_nn(X)
#         loss = loss_fn(yhat, y)
#
#         # Apply backprop
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch: {epoch} loss is {loss.item()}")
#
#     with open('model_state.pt', 'wb') as f:
#         save(my_nn.state_dict(), f)

# ===== LOAD MODEL and PREDICT =====
# with open('model_state.py', 'rb') as f:
#     my_nn.load_state_dict(load(f))
#     model_input = None
#     predict = torch.argmax(my_nn(model_input))
