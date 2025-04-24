"""
Train Transformer and compare to NTK/NTH kernel-based classifiers.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_model import SimpleTransformer
from nth_kernel import compute_ntk, compute_nth, kernel_regression, compute_gradients
from load_data import load_sst2

def train_transformer(model, dataloader, epochs=1, lr=2e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch["input_ids"])
            loss = F.cross_entropy(outputs, batch["label"])
            loss.backward()
            optimizer.step()
    print("Finished Transformer training")

def evaluate_ntk(model, dataset, n=100):
    inputs = dataset["input_ids"][:n]
    labels = dataset["label"][:n]
    ntk = compute_ntk(model, inputs, labels)
    predictions = ntk @ labels.float()
    predicted_labels = (predictions > predictions.mean()).long()
    acc = (predicted_labels == labels).float().mean().item()
    print("NTK Acc:", acc)


def run_kernel_eval(model, train_inputs, train_labels, test_inputs, test_labels):
    # NTK
    K_train_ntk, grads_train = compute_ntk(model, train_inputs, train_labels)
    grads_test = compute_gradients(model, test_inputs, test_labels)
    K_test_ntk = grads_test @ grads_train.T
    pred_ntk = kernel_regression(K_train_ntk, train_labels, K_test_ntk)
    pred_ntk_labels = (pred_ntk > 0.5).long()
    acc_ntk = (pred_ntk_labels == test_labels).float().mean().item()
    print(f"[NTK Test Accuracy] {acc_ntk:.4f}")

    # NTH
    K_train_nth, grads_train = compute_nth(model, train_inputs, train_labels)
    grads_test = []
    hess_test = []

    for i in range(len(test_inputs)):
        model.zero_grad()
        out = model(test_inputs[i].unsqueeze(0))
        loss = F.cross_entropy(out, test_labels[i].unsqueeze(0))
        g = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_g = torch.cat([p.view(-1) for p in g])
        grads_test.append(flat_g)
        dot = torch.sum(flat_g * flat_g)
        h = torch.autograd.grad(dot, model.parameters(), retain_graph=False)
        flat_h = torch.cat([h_.view(-1) for h_ in h])
        hess_test.append(flat_h)

    grads_test = torch.stack(grads_test)
    hess_test = torch.stack(hess_test)
    K_test_nth = grads_test @ grads_train.T + 0.5 * (hess_test @ grads_train.T)

    pred_nth = kernel_regression(K_train_nth, train_labels, K_test_nth)
    pred_nth_labels = (pred_nth > 0.5).long()
    acc_nth = (pred_nth_labels == test_labels).float().mean().item()
    print(f"[NTH Test Accuracy] {acc_nth:.4f}")

    return acc_ntk, acc_nth
