from data_utils import generate_data
from train_utils import train_network
from visualize import plot_losses

X, Y = generate_data()
widths = [5, 50, 500, 1000, 10000]
loss_results = {}

for width in widths:
    losses = train_network(width, X, Y)
    loss_results[width] = losses

plot_losses(loss_results)
