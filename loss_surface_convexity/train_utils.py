import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN

def train_network(width, X, Y, epochs=100, lr=0.001):
    model = SimpleNN(1, width, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
