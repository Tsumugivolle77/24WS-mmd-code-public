import jax.numpy as jnp
import numpy as np

if __name__ == "__main__":
    ds_path = "mmd_data_secret_polyxyz.npy"
    tsds = np.fromfile(ds_path).reshape((-1, 4))
    pass
    print(tsds)
