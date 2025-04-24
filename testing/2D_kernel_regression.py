import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
X = np.random.uniform(0, 1000, (100, 2))
y = np.sin(X[:, 0] / 100) + np.cos(X[:, 1] / 100)
gamma = 1e-5
K = rbf_kernel(X, X, gamma=gamma)
lambda_reg = 1e-2
alpha = np.linalg.solve(K, y)

def predict_kernel_regression(x_new, X_train, alpha, gamma):
    K_new = rbf_kernel(X_train, x_new.reshape(1, -1), gamma=gamma).flatten()
    return np.dot(K_new, alpha)


def train_nn(X, y, epochs=500, lr=1e-3):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def predict_nn(model, x):
    x_tensor = torch.tensor(x, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        return model(x_tensor).item()
model = train_nn(X, y)


x_test_1 = np.array([310.98, 325.18])
x_test_2 = np.array([789, 209])

f_test_1 = predict_kernel_regression(x_test_1, X, alpha, gamma)
f_test_2 = predict_kernel_regression(x_test_2, X, alpha, gamma)
nn_pred_1 = predict_nn(model, x_test_1)
nn_pred_2 = predict_nn(model, x_test_2)

print("\nKernel Regression Predictions:")
print("f(234, 789) =", f_test_1)
print("f(789, 209) =", f_test_2)

print("\nNeural Network Predictions:")
print("f(234, 789) =", nn_pred_1)
print("f(789, 209) =", nn_pred_2)