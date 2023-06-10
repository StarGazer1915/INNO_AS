import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from src.lmodel import Lmodel

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

my_nn = Lmodel().to('cuda')
optimizer = Adam(my_nn.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

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
