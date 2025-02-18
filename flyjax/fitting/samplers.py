import jax.numpy as jnp
import numpy as np

def base_randn_sampler(
    n_params: int,
    mean: float = 0.0,
    std: float = 1.0
) -> jnp.ndarray:
    return jnp.array(np.random.randn(n_params) * std + mean)

def base_uniform_sampler(
    n_params: int,
    low: float = -1.0,
    high: float = 1.0
) -> jnp.ndarray:
    return jnp.array(np.random.uniform(low, high, n_params))
