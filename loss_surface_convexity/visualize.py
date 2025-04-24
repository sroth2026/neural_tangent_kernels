import matplotlib.pyplot as plt

def plot_losses(loss_dict):
    plt.figure(figsize=(10, 6))
    for width, losses in loss_dict.items():
        plt.plot(losses, label=f"Width {width}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Convergence for Different Widths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_surface_comparison.png")
    plt.show()
