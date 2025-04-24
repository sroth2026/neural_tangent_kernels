# Neural Tangent Kernel vs Transformer: Training Dynamics and Interpretability

This project compares the training behavior and predictive performance of three different modeling paradigms:

- A small Transformer-based neural network trained via gradient descent
- Kernel regression using the Neural Tangent Kernel (NTK)
- Kernel regression using a second-order variant called the Neural Tangent Hierarchy (NTH)

The goal is to evaluate not only classification accuracy but also the **interpretability and training dynamics** of each model. This is especially relevant in low-data regimes where generalization, confidence calibration, and model structure matter as much as final performance.

---

## 📘 Overview

This project uses the **SST-2 sentiment classification dataset** (from the GLUE benchmark) and implements:

- A simple Transformer model in PyTorch
- Exact NTK and NTH computation using PyTorch autograd
- Kernel ridge regression using NTK/NTH feature representations
- Training visualization tools:
  - Loss curves
  - Gradient norms
  - Kernel matrix heatmaps
  - Confidence histograms

The implementation is focused on clarity, minimal assumptions, and reproducibility for educational or research use.

---

## 🔧 Project Structure

```text
transformers_ntk_nth/
│
├── main.py                # Entry point for training and evaluation
├── load_data.py           # Loads and tokenizes the SST-2 dataset
├── transformer_model.py   # A small Transformer for classification
├── train.py               # Training logic and NTK/NTH evaluation
├── nth_kernel.py          # Kernel construction using 1st and 2nd order gradients
├── visualize.py           # Kernel heatmap and confidence plotting tools
├── requirements.txt       # Python dependencies
└── README.md              # This file
