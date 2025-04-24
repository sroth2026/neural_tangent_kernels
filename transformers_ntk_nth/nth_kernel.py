"""
NTK and NTH kernel regression using PyTorch autograd.
"""

import torch

def compute_gradients(model, inputs, targets):
    """Compute flattened gradients of loss w.r.t. model params for each input."""
    grads = []
    for i in range(len(inputs)):
        model.zero_grad()
        out = model(inputs[i].unsqueeze(0))
        loss = torch.nn.functional.cross_entropy(out, targets[i].unsqueeze(0))
        grad = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False)
        grads.append(torch.cat([g.flatten() for g in grad]))
    return torch.stack(grads)  # [N, num_params]

def kernel_regression(K_train, y_train, K_test):
    """Kernel ridge regression prediction: f(x) = K_test @ (K_train + λI)^-1 y"""
    lam = 1e-4  # Regularization term
    K_reg = K_train + lam * torch.eye(K_train.size(0))
    alpha = torch.linalg.solve(K_reg, y_train.float())  # Solve (K + λI)^-1 y
    return K_test @ alpha  # Predict labels

def compute_ntk(model, inputs, targets):
    """Compute the NTK kernel matrix from gradients."""
    grads = compute_gradients(model, inputs, targets)
    K_train = grads @ grads.T
    return K_train, grads

def compute_nth(model, inputs, targets):
    """
    Approximate 2nd-order NTH kernel:
    Combines NTK (1st-order) and Hessian-induced kernel (2nd-order).
    """
    grads = []
    hess_feats = []

    for i in range(len(inputs)):
        model.zero_grad()
        out = model(inputs[i].unsqueeze(0))
        loss = torch.nn.functional.cross_entropy(out, targets[i].unsqueeze(0))
        g = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_g = torch.cat([p.view(-1) for p in g])
        grads.append(flat_g)

        # Second-order feature: directional derivative of grad along itself
        dot = torch.sum(flat_g * flat_g)
        hessian = torch.autograd.grad(dot, model.parameters(), retain_graph=False)
        flat_hess = torch.cat([h.view(-1) for h in hessian])
        hess_feats.append(flat_hess)

    grads = torch.stack(grads)
    hess_feats = torch.stack(hess_feats)

    # Combine NTK and 2nd-order kernel: K_total = K_grad + 0.5 * K_hess
    K_grad = grads @ grads.T
    K_hess = hess_feats @ hess_feats.T
    K_combined = K_grad + 0.5 * K_hess

    return K_combined, grads
