import numpy as np
import gymnasium as gym
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, InputLayer, Dropout, Activation
# from tensorflow.keras.layers import Conv2D, Flatten
from src.agent import Agent
from src.policy import Policy


# def create_model():
#     new_model = Sequential([
#         Dense(8),
#         Dense((128, 64)),
#         Dense(4, activation='relu'),
#     ])
#
#     return new_model.summary()


if __name__ == "__main__":
    # tf_model = create_model()
    tf_model = None

    epsilon = 1.0
    decay = 0.99

    p0 = Policy(tf_model, epsilon)
    a0 = Agent()

    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)

    for i in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"\n1.) observation: {list(observation)}\n2.) reward: {reward}\n"
              f"3.) available actions: {env.action_space}\n4.) performed action: {action}\n")
        if terminated or truncated:
            observation, info = env.reset()

        break

    env.close()
