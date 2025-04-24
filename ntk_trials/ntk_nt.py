import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from neural_tangents import stax

num_points = 100
angles = jnp.linspace(0, 2 * jnp.pi, num_points)
X = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
x0 = jnp.array([[0.0, -1.0]]) 
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512), stax.Relu(),
    stax.Dense(512), stax.Relu(),
    stax.Dense(1)
)
ntk_kernel = kernel_fn(x0, X, get='ntk')
ntk_values = ntk_kernel.reshape(-1)
plt.figure(figsize=(8, 5))
plt.plot(angles, ntk_values)
plt.title("NTK(x0, x) on the Unit Circle (Infinite-width Network)")
plt.xlabel("Angle (radians)")
plt.ylabel("NTK(x0, x)")
plt.grid(True)
plt.show()
