# flyjax/fitting/joint.py
import jax
import jax.numpy as jnp
import chex
import optax
from typing import List, Tuple, Callable, Optional, Dict, Union
from tqdm.auto import trange
from flyjax.fitting.evaluation import total_negative_log_likelihood

def total_nll_multi_group(
    theta_control: chex.Array,
    delta: chex.Array,
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_treatment: List[Tuple[chex.Array, chex.Array]],
    delta_penalty_sigma: float = 1.0,
) -> jnp.ndarray:
    """
    Compute the joint negative log likelihood for control and experimental data.

    The experimental group parameters are defined as:
         theta_exp = theta_control + delta.
    A quadratic penalty is applied on delta.

    Args:
        theta_control: Parameters for the control group.
        delta: Difference parameters.
        agent: The joint model function.
        experiments_control: List of experiments (choices, rewards) for the control group.
        experiments_treatment: List of experiments for the experimental group.
        delta_penalty_sigma: Standard deviation for the penalty.

    Returns:
        Total negative log likelihood (control + experimental + penalty).
    """
    nll_control = total_negative_log_likelihood(theta_control, agent, experiments_control)
    theta_exp = theta_control + delta
    nll_exp = total_negative_log_likelihood(theta_exp, agent, experiments_treatment)
    penalty = jnp.sum((delta / delta_penalty_sigma) ** 2) / 2.0
    return nll_control + nll_exp + penalty


def joint_train_model(
    init_theta_control: chex.Array,
    init_delta: chex.Array,
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_treatment: List[Tuple[chex.Array, chex.Array]],
    n_params: int = 4,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    delta_penalty_sigma: float = 1.0,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
    progress_bar: bool = True,
    return_history: str = "none",  # "none", "loss", or "full"
    callback: Optional[Callable[[int, chex.Array, float], None]] = None,
) -> Union[Tuple[chex.Array, chex.Array, bool],
           Tuple[Tuple[chex.Array, chex.Array, bool], Dict[str, List]]]:
    """
    Jointly train the control parameters and the difference delta using AdaBelief.

    Convergence is checked every 100 steps based on the relative change in loss.
    History tracking and callback options are provided similar to the single-group method.

    Args:
        init_theta_control: Initial guess for the control parameters.
        init_delta: Initial guess for the delta parameters.
        agent: The joint agent model (signature: (params, agent_state, choice, reward)).
        experiments_control: List of experiments for the control group.
        experiments_treatment: List of experiments for the experimental group.
        n_params: Number of parameters in theta_control.
        learning_rate: Learning rate (default 5e-2).
        num_steps: Maximum training steps (default 10,000).
        delta_penalty_sigma: Standard deviation for the quadratic penalty on delta.
        verbose: If True, prints loss information every 100 steps.
        early_stopping: Dictionary with keys "min_delta" and "patience" for early stopping.
        progress_bar: If True, shows a progress bar.
        return_history: "none" returns only final parameters and convergence flag;
                        "loss" returns a list of losses;
                        "full" returns a full history dictionary with "loss" and "params".
        callback: Optional callback function called each iteration with (step, params, loss).

    Returns:
        If return_history is "none": (theta_control_opt, delta_opt, converged_flag)
        If "loss": ((theta_control_opt, delta_opt, converged_flag), loss_history)
        If "full": ((theta_control_opt, delta_opt, converged_flag), full_history)
    """
    # Concatenate initial parameters.
    init_params = jnp.concatenate([init_theta_control, init_delta])
    optimizer = optax.adabelief(learning_rate)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def step_fn(params, opt_state):
        theta_control = params[:n_params]
        delta = params[n_params:]
        loss, grads = jax.value_and_grad(total_nll_multi_group, argnums=(0, 1))(
            theta_control, delta, agent, experiments_control, experiments_treatment, delta_penalty_sigma
        )
        grads_combined = jnp.concatenate([grads[0], grads[1].flatten()])
        updates, opt_state = optimizer.update(grads_combined, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    # Setup history tracking.
    history = {"loss": []}
    if return_history == "full":
        history["params"] = []

    best_loss = jnp.inf
    best_params = init_params
    patience_counter = 0
    convergence_threshold = (early_stopping.get("min_delta", 1e-2)
                             if early_stopping is not None else 1e-2)
    last_checkpoint_loss = None
    converged = False

    iterator = trange(num_steps, desc="Joint Training") if progress_bar else range(num_steps)
    params = init_params
    for i in iterator:
        params, opt_state, loss = step_fn(params, opt_state)
        loss_val = float(loss)
        history["loss"].append(loss_val)
        if return_history == "full":
            history["params"].append(params)

        if i % 100 == 0:
            if last_checkpoint_loss is not None:
                rel_change = abs((loss_val - last_checkpoint_loss) / last_checkpoint_loss)
                if rel_change < convergence_threshold:
                    if verbose:
                        print(f"Convergence reached at step {i} with relative change {rel_change:.4f}.")
                    best_params = params
                    converged = True
                    break
            last_checkpoint_loss = loss_val
            if verbose:
                print(f"Step {i:4d}, Joint NLL: {loss_val:.4f}")

        if callback is not None:
            callback(i, params, loss_val)

        if early_stopping is not None:
            if best_loss - loss_val > early_stopping.get("min_delta", 1e-4):
                best_loss = loss_val
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping.get("patience", 100):
                    if verbose:
                        print(f"Early stopping at step {i} with loss {loss_val:.4f}")
                    best_params = params
                    break

    theta_control_opt = params[:n_params]
    delta_opt = params[n_params:]
    if return_history == "full":
        return (theta_control_opt, delta_opt, converged), history
    elif return_history == "loss":
        return (theta_control_opt, delta_opt, converged), history["loss"]
    else:
        return theta_control_opt, delta_opt, converged


def multi_start_joint_train(
    n_restarts: int,
    init_theta_sampler: Callable[[], chex.Array],
    init_delta_sampler: Callable[[], chex.Array],
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_treatment: List[Tuple[chex.Array, chex.Array]],
    n_params: int = 4,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    delta_penalty_sigma: float = 1.0,
    min_num_converged: int = 3,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
    progress_bar: bool = True,
    get_history: bool = False,
) -> Union[Tuple[chex.Array, chex.Array, float],
           Tuple[Tuple[chex.Array, chex.Array, float], Dict[int, List]]]:
    """
    Run multiple training runs with different random initializations for the joint model.
    Stops early if at least min_num_converged runs have converged to the best loss.

    Args:
        n_restarts: Number of training runs.
        init_theta_sampler: Function to generate a new initial theta_control.
        init_delta_sampler: Function to generate a new initial delta.
        agent: The joint agent model.
        experiments_control: Dataset for the control group.
        experiments_treatment: Dataset for the treatment group.
        n_params: Number of parameters in theta_control.
        learning_rate: Learning rate.
        num_steps: Maximum training steps per run.
        delta_penalty_sigma: Std for the penalty.
        min_num_converged: Minimum number of converged runs before stopping.
        verbose: If True, prints progress information.
        early_stopping: Dictionary with early stopping parameters.
        progress_bar: If True, shows a progress bar.
        get_history: If True, return loss histories for each run.

    Returns:
        If get_history is False:
            (best_theta_control, best_delta, best_loss)
        If get_history is True:
            ((best_theta_control, best_delta, best_loss), loss_history)
    """
    best_theta = None
    best_delta = None
    best_loss = jnp.inf
    loss_history = {}
    num_converged = 0

    for i in range(n_restarts):
        print(f"\n--- Joint Restart {i+1}/{n_restarts} ---")
        init_theta = init_theta_sampler()
        init_delta = init_delta_sampler()
        if get_history:
            (theta_opt, delta_opt, converged), history = joint_train_model(
                init_theta,
                init_delta,
                agent,
                experiments_control,
                experiments_treatment,
                n_params=n_params,
                learning_rate=learning_rate,
                num_steps=num_steps,
                delta_penalty_sigma=delta_penalty_sigma,
                verbose=verbose,
                early_stopping=early_stopping,
                progress_bar=progress_bar,
                return_history="loss",
            )
            loss_history[i] = history
        else:
            theta_opt, delta_opt, converged = joint_train_model(
                init_theta,
                init_delta,
                agent,
                experiments_control,
                experiments_treatment,
                n_params=n_params,
                learning_rate=learning_rate,
                num_steps=num_steps,
                delta_penalty_sigma=delta_penalty_sigma,
                verbose=verbose,
                early_stopping=early_stopping,
                progress_bar=progress_bar,
                return_history="none",
            )
        current_loss = total_nll_multi_group(theta_opt, delta_opt, agent,
                                               experiments_control,
                                               experiments_treatment,
                                               delta_penalty_sigma)
        print(f"Restart {i+1} final Joint NLL: {current_loss:.4f}")
        if converged:
            num_converged += 1
        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta_opt
            best_delta = delta_opt
        if num_converged >= min_num_converged:
            print(f"Stopping early because {min_num_converged} runs have converged to the current best loss.")
            break

    print(f"\nBest Joint NLL: {best_loss:.4f}")
    if get_history:
        return (best_theta, best_delta, best_loss), loss_history
    else:
        return best_theta, best_delta, best_loss


def evaluate_joint_model(
    theta_control: chex.Array,
    delta: chex.Array,
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_treatment: List[Tuple[chex.Array, chex.Array]],
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
    nll_exp = total_negative_log_likelihood(theta_exp, agent, experiments_treatment)
    penalty = jnp.sum((delta / delta_penalty_sigma) ** 2) / 2.0
    joint_nll = nll_control + nll_exp + penalty
    return float(nll_control), float(nll_exp), float(joint_nll)
