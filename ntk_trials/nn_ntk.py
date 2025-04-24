import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    X_train = train_dataset.data.numpy().reshape(-1, 28*28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28*28) / 255.0
    y_test = test_dataset.targets.numpy()
    return X_train, y_train, X_test, y_test

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

def train_nn(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    train_preds_over_time = []
    X_track_tensor = torch.tensor(X_val[:200], dtype=torch.float32).to(device)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices].to(device), y_train_tensor[indices].to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_preds = torch.argmax(val_outputs, dim=1)
            acc = accuracy_score(y_val_tensor.cpu(), val_preds.cpu())
            preds = model(X_track_tensor).cpu().numpy()
            train_preds_over_time.append(preds)
            print(f'Epoch {epoch+1}: Validation Accuracy: {acc:.4f}')
    return model, train_preds_over_time




def relu_ntk(x, xp):
    norm_x = np.linalg.norm(x)
    norm_xp = np.linalg.norm(xp)
    cos_theta = np.dot(x, xp) / (norm_x * norm_xp + 1e-10)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return (norm_x * norm_xp) / np.pi * (np.sin(theta) + (np.pi - theta) * cos_theta)

def compute_ntk_matrix(X1, X2=None):
    if X2 is None:
        X2 = X1
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = relu_ntk(X1[i], X2[j])
    return K

def ntk_predict(X_train, y_train, X_test, lam=1e-3):
    K_train = compute_ntk_matrix(X_train)
    num_classes = 10
    Y = np.eye(num_classes)[y_train]
    alpha = np.linalg.solve(K_train + lam * np.eye(K_train.shape[0]), Y)
    K_test = compute_ntk_matrix(X_test, X_train)
    Y_pred = np.dot(K_test, alpha)
    y_pred = np.argmax(Y_pred, axis=1)
    return y_pred


def main():
    X_train, y_train, X_test, y_test = load_mnist()
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = SimpleNN()
    model, train_preds_over_time = train_nn(model, X_tr, y_tr, X_val, y_val, epochs=5)
    X_train_ntk = X_tr[:1000]
    y_train_ntk = y_tr[:1000]
    X_track = X_val[:200]
    y_track = y_val[:200]
    ntk_preds = ntk_predict(X_train_ntk, y_train_ntk, X_track)
    ntk_acc = accuracy_score(y_track, ntk_preds)
    nn_accs = []
    for epoch_preds in train_preds_over_time:
        y_pred_nn = np.argmax(epoch_preds, axis=1)
        acc = accuracy_score(y_track, y_pred_nn)
        nn_accs.append(acc)
    plt.plot(range(1, len(nn_accs)+1), nn_accs, label="NN Accuracy Over Epochs")
    plt.hlines(ntk_acc, 1, len(nn_accs), colors='r', linestyles='dashed', label="NTK Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy on Tracked Subset")
    plt.title("Neural Network vs NTK Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()