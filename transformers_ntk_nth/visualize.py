import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_kernel_heatmap(K, title="Kernel Matrix", size=8):
    plt.figure(figsize=(size, size))
    sns.heatmap(K.detach().cpu().numpy(), cmap="viridis", square=True)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.show()
