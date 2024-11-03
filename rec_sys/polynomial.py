# implement and test Exercise 3
from itertools import product
import jax
import jax.numpy as jnp
import numpy as np
import random

from matplotlib import pyplot as plt
from tensorflow.python.ops.numpy_ops.np_dtypes import float32, float64

from rec_sys.jax_intro import n_points


# %% Generate polynomial for exercise 4
def polynomial_generate(Ns: list[3], terms: int) -> dict[tuple, jnp.array]:
    # generate exactly t coeffs
    rnd_seed = 114514
    rnd_key  = jax.random.PRNGKey(rnd_seed)
    coeffs   = jax.random.normal(rnd_key, (terms,))
    p        = dict()
    ensured  = 0

    def get_rand_tup():
        while True:
            # There's no (simple) way to convert jax array
            # into hashable objects. Choose this ugly
            # way w\ built-in `random.randint` instead :-(
            i = random.randint(0, Ns[0])
            j = random.randint(0, Ns[1])
            k = random.randint(0, Ns[2])
            if (i, j, k) not in p.keys():
                return i, j, k

    # ensure max deg
    flg = [False, False, False]
    all_max_deg_possibilities = \
      [(i, j, k) for i, j, k in product(range(1, Ns[0]+1), range(1, Ns[1]+1), range(1, Ns[2]+1))
       if i == Ns[0] or j == Ns[1] or k == Ns[2]]

    while not all(flg):
        if ensured + 1 == terms:
            not_ensured_var = (index for index in range(0, 3) if index == False)
            p[not_ensured_var] = coeffs[ensured]
            ensured += 1

        tup = all_max_deg_possibilities[random.randint(0, len(all_max_deg_possibilities)) - 1]

        if tup in p.keys():
            continue
        # else
        p[tup] = coeffs[ensured]
        ensured += 1
        for i in range(0, 3):
            if tup[i] == Ns[i]:
                flg[i] = True

    # generate the remaining terms
    for term in range(ensured, terms):
        tup = get_rand_tup()
        p[tup] = coeffs[term]

    return p

p = polynomial_generate([2, 2, 1], 5)
print(p)

def f(x, y, z, w):
    return sum([p[key] * (pow(x, key[0]) - w) * (pow(y, key[1]) + 2 * w) * pow(z, key[2]) for key in p.keys()])

# Generate data
# n_points, noise_frac, rnd_seed = 1000, 0.25, 42
# x = jnp.linspace(-3, 6, n_points)
# y = jnp.linspace(0, 12, n_points)
# z = jnp.linspace(1, 20, n_points)
# out_pure = f(x, y, z, 2.0)
# # Add some noise to data
# rnd_key = jax.random.PRNGKey(rnd_seed)
# out_with_noise = out_pure + out_pure * noise_frac * jax.random.normal(rnd_key, (n_points,))
# # Stack x and y_with_noise into a single array
# train_ds = jnp.stack((x, y, z, out_with_noise), axis=1)
# print(f"Training data (first 5):\n {train_ds[:5]}")

# %% Read data
ds_path = "/Users/offensive77/Code/24WS-mmd-code-public-forked/rec_sys/mmd_data_secret_polyxyz.npy"
train_ds = jnp.asarray(np.fromfile(ds_path).reshape((-1, 4)), float)
n_points = train_ds.shape[0]

# %% Define a simple loss function and its gradient
def loss(param_w, data):
    # return  jnp.sum((data[:,1] - f(data[:,0], param_w))**2)
    return jnp.log(jnp.sum((data[:, 3] - f(data[:, 0], data[:, 1], data[:, 2], param_w)) ** 2))


# Using JAX automatic differentiation - autograd
grad_loss = jax.grad(loss)


# %% Note that grad_loss is a function!
param_w = 1.0
print(f"\n\nLoss value for param_w = {param_w}: {loss(param_w, train_ds)}\n\n")
print(jax.make_jaxpr(grad_loss)(param_w, train_ds))


# %% Plot the loss function and its gradient
def compute_loss_and_grad(param_w, data, start, stop, num_points=100):
    param_w_values = jnp.linspace(start, stop, num_points)
    loss_values = jnp.array([loss(w, data) for w in param_w_values])
    grad_values = jnp.array([grad_loss(w, data) for w in param_w_values])
    return param_w_values, loss_values, grad_values


param_w_values, loss_values, grad_values = (
    compute_loss_and_grad(0.0, train_ds, -3, 10))

# plt.plot(param_w_values, loss_values, label='Loss')
# plt.plot(param_w_values, grad_values, label='Gradient')
# plt.legend()
# plt.show()


# %% Run the SG loop
num_epochs = 300
learning_rate = 0.005
param_w = 0.0  # Initial guess for the parameter

print("\n===== Running Gradient Descent =====")
for epoch in range(num_epochs):
    grad = grad_loss(param_w, train_ds)
    param_w = param_w - learning_rate * grad
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: param_w={param_w}, grad={grad}, loss={loss(param_w, train_ds)}")


# %% Plot the results
# plt.plot(x, y, z, out_pure, label='True function')
# plt.plot(x, y, z, out_with_noise, 'o', label='Data with noise')
# plt.plot(x, y, z, f(x, y, z, param_w), label='Fitted function')
# plt.legend()
# plt.show()


# %% Run stochastic gradient descent
num_epochs = 50
learning_rate = 0.01
param_w = 0.0
num_points_per_batch = n_points // 5
print("\n===== Running Stochastic Gradient Descent =====")
for epoch in range(num_epochs):
    # Get points for the current batch
    for i in range(0, n_points, num_points_per_batch):
        batch = train_ds[i:i + num_points_per_batch]
        grad = grad_loss(param_w, batch)
        param_w = param_w - learning_rate * grad

    print(f"Epoch {epoch}: param_w={param_w}, grad={grad}, loss={loss(param_w, train_ds)}")
