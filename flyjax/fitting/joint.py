import jax
import jax.numpy as jnp
import chex
import optax
from typing import List, Tuple, Callable, Optional, Dict, Union
from tqdm.auto import trange
from flyjax.fitting.evaluation import total_negative_log_likelihood


def total_negative_log_likelihood_multi_group(
    theta_control: chex.Array,
    delta: chex.Array,
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_exp: List[Tuple[chex.Array, chex.Array]],
    delta_penalty_sigma: float = 1.0
) -> jnp.ndarray:
    """
    Compute the joint negative log likelihood for control and experimental data.
    
    The experimental group parameters are defined as:
         theta_exp = theta_control + delta.
    
    A quadratic penalty (equivalent to a Gaussian prior) is applied on delta.
    
    Args:
        theta_control: Parameters for the control group.
        delta: Difference parameters, so that experimental parameters are theta_control + delta.
        agent: The agent model function that takes parameters and agent state and returns
        experiments_control: List of (choices, rewards) for the control group.
        experiments_exp: List of (choices, rewards) for the experimental group.
        delta_penalty_sigma: Standard deviation of the Gaussian prior on delta.
        
    Returns:
        Total negative log likelihood (control + experimental + penalty).
    """
    # Compute NLL for control data.
    nll_control = total_negative_log_likelihood(theta_control, agent, experiments_control)
    # Compute experimental parameters and its NLL.
    theta_exp = theta_control + delta
    nll_exp = total_negative_log_likelihood(theta_exp, agent, experiments_exp)
    # Quadratic penalty on delta (MAP equivalent to a zero-mean Gaussian prior).
    penalty = jnp.sum((delta / delta_penalty_sigma)**2) / 2.0
    return nll_control + nll_exp + penalty


def joint_train_model(
    init_theta_control: chex.Array,
    init_delta: chex.Array,
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_exp: List[Tuple[chex.Array, chex.Array]],
    n_params: int = 4,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    delta_penalty_sigma: float = 1.0,
    verbose: bool = True
) -> Tuple[chex.Array, chex.Array]:
    """
    Jointly train control parameters and the difference delta.
    
    Returns:
        Optimized theta_control and delta.
    """
    # Stack initial parameters: first n_params entries for theta_control, next n_params for delta.
    init_params = jnp.concatenate([init_theta_control, init_delta])
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)
    
    @jax.jit
    def step_fn(params, opt_state):
        # Unpack parameters.
        theta_control = params[:n_params]
        delta = params[n_params:]
        # Use argnums=(0,1) so that gradients for theta_control and delta are computed.
        loss, grads = jax.value_and_grad(total_negative_log_likelihood_multi_group, argnums=(0,1))(
            theta_control, delta, agent, experiments_control, experiments_exp, delta_penalty_sigma)
        # Concatenate the two gradients to form one vector of shape (2 * n_params,).
        grads_combined = jnp.concatenate(grads)
        updates, opt_state = optimizer.update(grads_combined, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, loss, opt_state

    params = init_params
    for i in trange(num_steps, desc="Joint Training"):
        params, loss, opt_state = step_fn(params, opt_state)
        if verbose and (i % 100 == 0):
            print(f"Step {i:4d}, Joint NLL: {loss:.4f}")
    
    theta_control_opt = params[:n_params]
    delta_opt = params[n_params:]
    return theta_control_opt, delta_opt


def multi_start_joint_train(
    n_restarts: int,
    init_theta_sampler: Callable[[], chex.Array],
    init_delta_sampler: Callable[[], chex.Array],
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_exp: List[Tuple[chex.Array, chex.Array]],
    n_params: int = 4,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    delta_penalty_sigma: float = 1.0,
    verbose: bool = True
) -> Tuple[chex.Array, chex.Array, List[float]]:
    """
    Run multiple training runs (with different random initializations) for the joint model.
    
    Returns:
        Best theta_control, best delta, and a list of final joint losses for each restart.
    """
    best_theta = None
    best_delta = None
    best_loss = jnp.inf
    all_losses = []
    
    for i in range(n_restarts):
        print(f"\n--- Joint Restart {i+1}/{n_restarts} ---")
        init_theta = init_theta_sampler()
        init_delta = init_delta_sampler()
        theta_opt, delta_opt = joint_train_model(
            init_theta, init_delta, agent,
            experiments_control, experiments_exp,
            n_params=n_params,
            learning_rate=learning_rate,
            num_steps=num_steps,
            delta_penalty_sigma=delta_penalty_sigma,
            verbose=verbose
        )
        # Evaluate joint NLL for these parameters.
        joint_nll = total_negative_log_likelihood_multi_group(
            theta_opt, delta_opt, agent, experiments_control, experiments_exp, delta_penalty_sigma)
        print(f"Restart {i+1} final Joint NLL: {joint_nll:.4f}")
        all_losses.append(float(joint_nll))
        if joint_nll < best_loss:
            best_loss = joint_nll
            best_theta = theta_opt
            best_delta = delta_opt
            
    print(f"\nBest Joint NLL: {best_loss:.4f}")
    return best_theta, best_delta, all_losses

def evaluate_joint_model(
    theta_control: chex.Array,
    delta: chex.Array,
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_exp: List[Tuple[chex.Array, chex.Array]],
    delta_penalty_sigma: float = 1.0
) -> Tuple[float, float, float]:
    """
    Evaluate the joint model by computing NLLs for control data, experimental data,
    and the total joint NLL.
    
    Returns:
        (nll_control, nll_exp, joint_nll)
    """
    nll_control = total_negative_log_likelihood(theta_control, agent, experiments_control)
    theta_exp = theta_control + delta
    nll_exp = total_negative_log_likelihood(theta_exp, agent, experiments_exp)
    penalty = jnp.sum((delta / delta_penalty_sigma)**2) / 2.0
    joint_nll = nll_control + nll_exp + penalty
    return float(nll_control), float(nll_exp), float(joint_nll)