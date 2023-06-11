import gymnasium as gym


class Agent:
    def __init__(self, memory, policy):
        self.memory = memory
        self.policy = policy

    def train(self, num_epochs, env):
        observation, info = env.reset(seed=42)

        for i in range(num_epochs):
            action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"\n1.) observation: {list(observation)}\n2.) reward: {reward}\n"
                  f"3.) available actions: {env.action_space}\n4.) performed action: {action}\n")
            if terminated or truncated:
                observation, info = env.reset()

            break

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
