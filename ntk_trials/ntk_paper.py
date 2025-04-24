import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Experiment settings
n_points = 100
angles = np.linspace(0, 2 * np.pi, n_points)
X = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # Points on unit circle
x0 = np.array([[1.0, 0.0]])  # Fixed point x0

# Define a simple fully connected ReLU network
class SimpleReLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(2, width)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Compute empirical NTK using autograd
def compute_empirical_ntk(model, x1, x2):
    ntk = torch.zeros((x1.size(0), x2.size(0)))
    for i in range(x1.size(0)):
        model.zero_grad()
        out1 = model(x1[i:i+1])
        grad1 = torch.autograd.grad(out1, model.parameters(), retain_graph=True, create_graph=True)
        grad1 = torch.cat([g.view(-1) for g in grad1])

        for j in range(x2.size(0)):
            model.zero_grad()
            out2 = model(x2[j:j+1])
            grad2 = torch.autograd.grad(out2, model.parameters(), retain_graph=True, create_graph=True)
            grad2 = torch.cat([g.view(-1) for g in grad2])

            ntk[i, j] = torch.dot(grad1, grad2)
    return ntk.detach().cpu().numpy()

# Run experiment for different widths
def experiment_ntk_on_circle(widths, n_trials=3):
    results = {}
    x0_torch = torch.tensor(x0, dtype=torch.float32)
    X_torch = torch.tensor(X, dtype=torch.float32)

    for width in widths:
        ntks = []
        for _ in range(n_trials):
            model = SimpleReLU(width)
            ntk = compute_empirical_ntk(model, x0_torch, X_torch)[0]  # NTK(x0, x) for all x
            ntks.append(ntk)
        ntks = np.stack(ntks)
        results[width] = {
            'mean': np.mean(ntks, axis=0),
            'std': np.std(ntks, axis=0)
        }
    return results

# Plot results
widths_to_try = [50, 200, 1000]
results = experiment_ntk_on_circle(widths_to_try, n_trials=5)

plt.figure(figsize=(10, 6))
for width in widths_to_try:
    plt.plot(angles, results[width]['mean'], label=f'Width {width}')
    plt.fill_between(angles,
                     results[width]['mean'] - results[width]['std'],
                     results[width]['mean'] + results[width]['std'],
                     alpha=0.2)

plt.title("NTK(x0, x) along Unit Circle for Various Network Widths")
plt.xlabel("Angle (radians)")
plt.ylabel("NTK(x0, x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()