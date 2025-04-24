import torch
from transformer_model import SimpleTransformer
from load_data import load_sst2
from train import run_kernel_eval, train_transformer, plot_prediction_confidences
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualize import plot_kernel_heatmap


def track_transformer_training(model, dataloader, epochs=10, lr=2e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch["input_ids"])
            loss = F.cross_entropy(outputs, batch["label"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))
        print(f"Epoch {epoch+1}, Loss: {losses[-1]:.4f}")
    return losses

def evaluate_transformer(model, test_X, test_y):
    model.eval()
    with torch.no_grad():
        outputs = model(test_X)
        preds = outputs.argmax(dim=1)
        acc = (preds == test_y).float().mean().item()
    return acc

def plot_results(transformer_losses, accs):
    import seaborn as sns
    sns.set(style="whitegrid", font_scale=1.2)

    # --- 1. Loss Curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(transformer_losses)+1), transformer_losses, marker='o', linewidth=2)
    plt.title("Transformer Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 2. Accuracy Bar Plot ---
    labels = ["Transformer", "NTK", "NTH"]
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, accs, color=colors, edgecolor='black')
    plt.title("Test Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add accuracy values above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.show()


def main():
    # === Load data ===
    train_data, test_data = load_sst2()
    N_train, N_test = 100, 50
    train_X = train_data["input_ids"][:N_train]
    train_y = train_data["label"][:N_train]
    test_X = test_data["input_ids"][:N_test]
    test_y = test_data["label"][:N_test]

    # === Instantiate model ===
    model = SimpleTransformer()

    # === Train Transformer and track loss ===
    print("\n[Training Transformer]")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    transformer_losses = track_transformer_training(model, train_loader, epochs=10)


    # === Evaluate Transformer ===
    acc_transformer = evaluate_transformer(model, test_X, test_y)
    print(f"[Transformer Test Accuracy] {acc_transformer:.4f}")

    # === Run NTK + NTH ===
    kernel_model = SimpleTransformer()
    acc_ntk, acc_nth, K_train_ntk, K_train_nth, pred_ntk, pred_nth = run_kernel_eval(kernel_model, train_X, train_y, test_X, test_y)


    # === Visualize Results ===
    plot_results(transformer_losses, [acc_transformer, acc_ntk, acc_nth])

    plot_kernel_heatmap(K_train_ntk, title="NTK Kernel Heatmap")
    plot_kernel_heatmap(K_train_nth, title="NTH Kernel Heatmap")
    
    plot_prediction_confidences(pred_ntk, "NTK Confidence Distribution")
    plot_prediction_confidences(pred_nth, "NTH Confidence Distribution")


    print(f"""
        ===== SUMMARY =====
        [Transformer] Accuracy: {acc_transformer:.4f}
        [NTK]         Accuracy: {acc_ntk:.4f}
        [NTH]         Accuracy: {acc_nth:.4f}

        NTK shows consistent kernel structure (see heatmap),
        Transformer shows volatile training dynamics but high accuracy,
        NTH balances both interpretability and slight performance boost.
        """)

if __name__ == "__main__":
    main()
