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
    delta_penalty_sigma: float = 1.0,
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
    nll_control = total_negative_log_likelihood(
        theta_control, agent, experiments_control
    )
    # Compute experimental parameters and its NLL.
    theta_exp = theta_control + delta
    nll_exp = total_negative_log_likelihood(theta_exp, agent, experiments_exp)
    # Quadratic penalty on delta (MAP equivalent to a zero-mean Gaussian prior).
    penalty = jnp.sum((delta / delta_penalty_sigma) ** 2) / 2.0
    return nll_control + nll_exp + penalty


def joint_train_model(
    init_theta_control: chex.Array,
    init_delta: chex.Array,
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_exp: List[Tuple[chex.Array, chex.Array]],
    n_params: int = 4,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,  # increased from 1000
    delta_penalty_sigma: float = 1.0,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
) -> Tuple[chex.Array, chex.Array, bool]:
    """
    Jointly train control parameters and the difference delta.

    Uses AdaBelief with learning rate 5e-2 and runs up to 10,000 steps unless convergence is detected.
    Convergence is checked every 100 steps: if the relative change in loss is below the threshold
    (default 1e-2, or as provided in early_stopping), training stops early.

    Returns:
        Optimized theta_control, optimized delta, and a converged_flag.
    """
    # Stack initial parameters: first n_params for theta_control, next for delta.
    init_params = jnp.concatenate([init_theta_control, init_delta])
    optimizer = optax.adabelief(learning_rate)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def step_fn(params, opt_state):
        theta_control = params[:n_params]
        delta = params[n_params:]
        loss, grads = jax.value_and_grad(
            total_negative_log_likelihood_multi_group, argnums=(0, 1)
        )(
            theta_control,
            delta,
            agent,
            experiments_control,
            experiments_exp,
            delta_penalty_sigma,
        )
        grads_combined = jnp.concatenate([grads[0], grads[1].flatten()])
        updates, opt_state = optimizer.update(grads_combined, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, loss, opt_state

    params = init_params
    last_checkpoint_loss = None
    converged = False
    iterator = (
        trange(num_steps, desc="Joint Training") if progress_bar else range(num_steps)
    )

    for i in iterator:
        params, loss, opt_state = step_fn(params, opt_state)
        loss_val = float(loss)

        # Every 100 steps, check convergence.
        if i % 100 == 0:
            if last_checkpoint_loss is not None:
                rel_change = abs(
                    (loss_val - last_checkpoint_loss) / last_checkpoint_loss
                )
                if (
                    rel_change < early_stopping.get("min_delta", 1e-2)
                    if early_stopping
                    else 1e-2
                ):
                    if verbose:
                        print(
                            f"Convergence reached at step {i} with relative change {rel_change:.4f}."
                        )
                    converged = True
                    break
            last_checkpoint_loss = loss_val

            if verbose:
                print(f"Step {i:4d}, Joint NLL: {loss_val:.4f}")

    theta_control_opt = params[:n_params]
    delta_opt = params[n_params:]
    return theta_control_opt, delta_opt, converged


def multi_start_joint_train(
    n_restarts: int,
    init_theta_sampler: Callable[[], chex.Array],
    init_delta_sampler: Callable[[], chex.Array],
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_exp: List[Tuple[chex.Array, chex.Array]],
    n_params: int = 4,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,  # increased from 1000
    delta_penalty_sigma: float = 1.0,
    min_num_converged: int = 3,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
) -> Tuple[chex.Array, chex.Array, float]:
    """
    Run multiple training runs with different random initializations for the joint model.

    The process stops early if at least `min_num_converged` runs converge to the current best loss.

    Args:
        n_restarts: Number of training runs.
        init_theta_sampler: Function returning a new initial theta_control array.
        init_delta_sampler: Function returning a new initial delta array.
        agent: The agent model function.
        experiments_control: Dataset for the control group.
        experiments_exp: Dataset for the experimental group.
        learning_rate: Learning rate (default 5e-2).
        num_steps: Maximum training steps per run (default 10,000).
        delta_penalty_sigma: Standard deviation for the penalty.
        min_num_converged: Minimum number of runs that must converge to the best loss.
        verbose: If True, prints progress information.
        early_stopping: Dictionary with early stopping parameters.

    Returns:
        A tuple (best_theta_control, best_delta, best_loss), where best_loss is the best joint NLL.
    """
    best_theta = None
    best_delta = None
    best_loss = jnp.inf
    all_losses = []
    num_converged = 0

    for i in range(n_restarts):
        print(f"\n--- Joint Restart {i+1}/{n_restarts} ---")
        init_theta = init_theta_sampler()
        init_delta = init_delta_sampler()
        theta_opt, delta_opt, converged = joint_train_model(
            init_theta,
            init_delta,
            agent,
            experiments_control,
            experiments_exp,
            n_params=n_params,
            learning_rate=learning_rate,
            num_steps=num_steps,
            delta_penalty_sigma=delta_penalty_sigma,
            verbose=verbose,
            early_stopping=early_stopping,
        )
        joint_nll = total_negative_log_likelihood_multi_group(
            theta_opt,
            delta_opt,
            agent,
            experiments_control,
            experiments_exp,
            delta_penalty_sigma,
        )
        print(f"Restart {i+1} final Joint NLL: {joint_nll:.4f}")
        all_losses.append(float(joint_nll))
        if converged:
            num_converged += 1
        if joint_nll < best_loss:
            best_loss = joint_nll
            best_theta = theta_opt
            best_delta = delta_opt
        if num_converged >= min_num_converged:
            print(
                f"Stopping early because {min_num_converged} runs have converged to the current best loss."
            )
            break

    print(f"\nBest Joint NLL: {best_loss:.4f}")
    return best_theta, best_delta, best_loss


def evaluate_joint_model(
    theta_control: chex.Array,
    delta: chex.Array,
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_exp: List[Tuple[chex.Array, chex.Array]],
    delta_penalty_sigma: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Evaluate the joint model by computing NLLs for control data, experimental data,
    and the total joint NLL.

    Returns:
        (nll_control, nll_exp, joint_nll)
    """
    nll_control = total_negative_log_likelihood(
        theta_control, agent, experiments_control
    )
    theta_exp = theta_control + delta
    nll_exp = total_negative_log_likelihood(theta_exp, agent, experiments_exp)
    penalty = jnp.sum((delta / delta_penalty_sigma) ** 2) / 2.0
    joint_nll = nll_control + nll_exp + penalty
    return float(nll_control), float(nll_exp), float(joint_nll)
