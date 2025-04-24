import jax.numpy as jnp
import jax
from jax import random
from jax.example_libraries import stax
from neural_tangents import stax as nt_stax
from neural_tangents import predict
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist['data'] / 255.0
y = mnist['target'].astype(int)

# Subsample to speed things up
X, y = X[:1000], y[:1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train_oh = jax.nn.one_hot(y_train, 10)

# 2. Define infinite-width NTK model
init_fn, apply_fn, kernel_fn = nt_stax.serial(
    nt_stax.Dense(1024), nt_stax.Relu(),
    nt_stax.Dense(1024), nt_stax.Relu(),
    nt_stax.Dense(10)
)

# 3. Compute kernel regression predictor
print("Computing NTK predictions...")
predict_fn = predict.gradient_descent_mse_ensemble(kernel_fn, X_train, y_train_oh)
y_test_ntk = predict_fn(x_test=X_test, get='ntk')
y_pred_ntk = jnp.argmax(y_test_ntk, axis=1)

# 4. Accuracy of NTK model
ntk_acc = jnp.mean(y_pred_ntk == y_test)
print(f"NTK Test Accuracy: {ntk_acc:.4f}")

# 5. (Optional) Plot prediction probabilities for some test digits
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.bar(range(10), y_test_ntk[i])
    plt.title(f"True: {y_test[i]}")
    plt.xticks([])
    plt.yticks([])
plt.suptitle("NTK Prediction Distributions on MNIST Test Samples")
plt.tight_layout()
plt.show()
