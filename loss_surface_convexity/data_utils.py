import numpy as np
import torch

def generate_data(n_samples=100):
    X = np.linspace(-1, 1, n_samples).reshape(-1, 1)
    Y = np.sin(np.pi * X)
    Y = Y / np.max(np.abs(Y))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
