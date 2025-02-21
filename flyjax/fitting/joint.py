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
    Compute the joint negative log likelihood (NLL) for two groups by combining 
    the control group's likelihood, the treatment group's likelihood (using an offset), 
    and a quadratic penalty on the delta parameter.

    The treatment group's parameters are computed as:
        theta_exp = theta_control + delta

    :param theta_control: Parameter array for the control group.
    :type theta_control: chex.Array
    :param delta: Offset parameters to obtain treatment parameters.
    :type delta: chex.Array
    :param agent: Model function for likelihood evaluation.
    :type agent: Callable
    :param experiments_control: List of experiments with (choices, rewards) for control.
    :type experiments_control: List[Tuple[chex.Array, chex.Array]]
    :param experiments_treatment: List of experiments for the treatment group.
    :type experiments_treatment: List[Tuple[chex.Array, chex.Array]]
    :param delta_penalty_sigma: Standard deviation scale for the quadratic penalty on delta.
    :type delta_penalty_sigma: float

    :returns: Total joint NLL including the penalty.
    :rtype: jnp.ndarray

    :note: This function aggregates NLLs from both groups and adds a regularization term
           to discourage large deltas.
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
    n_params: int,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    delta_penalty_sigma: float = 1.0,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
    progress_bar: bool = True,
    return_history: str = "none",  # "none", "loss", or "full"
    callback: Optional[Callable[[int, chex.Array, float], None]] = None,
) -> Union[Tuple[chex.Array, chex.Array, bool],
           Tuple[chex.Array, chex.Array, List[float], bool],
           Tuple[chex.Array, chex.Array, Dict[str, List[float]], bool]]:
    """
    Jointly train control parameters and the delta offset using AdaBelief optimization.

    This function concatenates the control and delta parameters, then iteratively
    updates them using gradient-based steps. The joint NLL is computed from control data,
    treatment data (with offset applied), and a quadratic penalty on delta. Convergence is
    checked every 100 steps based on the relative change in loss, with an optional early
    stopping mechanism and callback support.

    :param init_theta_control: Initial parameter array for the control group.
    :type init_theta_control: chex.Array
    :param init_delta: Initial delta offset array.
    :type init_delta: chex.Array
    :param agent: Joint model function for likelihood evaluation.
    :type agent: Callable
    :param experiments_control: List of experiments (choices, rewards) for the control group.
    :type experiments_control: List[Tuple[chex.Array, chex.Array]]
    :param experiments_treatment: List of experiments for the treatment group.
    :type experiments_treatment: List[Tuple[chex.Array, chex.Array]]
    :param n_params: Size of theta_control parameters (used to split the concatenated array).
    :type n_params: int
    :param learning_rate: Optimizer learning rate (default: 5e-2).
    :type learning_rate: float
    :param num_steps: Maximum number of training iterations (default: 10000).
    :type num_steps: int
    :param delta_penalty_sigma: Penalty scaling factor for delta.
    :type delta_penalty_sigma: float
    :param verbose: Flag to print loss updates every 100 steps.
    :type verbose: bool
    :param early_stopping: Dict with "patience" and "min_delta" to determine early stopping.
    :type early_stopping: Optional[Dict[str, float]]
    :param progress_bar: If True, displays a tqdm progress bar.
    :type progress_bar: bool
    :param return_history: Specifies return format: "none" for final params,
                           "loss" for loss history, "full" for complete history.
    :type return_history: str
    :param callback: Optional function called each iteration with (step, params, loss).
    :type callback: Optional[Callable[[int, chex.Array, float], None]]

    :returns: Depending on return_history:
              - "none": (theta_control_opt, delta_opt, converged_flag)
              - "loss": (theta_control_opt, delta_opt, loss_history, converged_flag)
              - "full": (theta_control_opt, delta_opt, full_history, converged_flag)
    :rtype: Union[Tuple[chex.Array, chex.Array, bool],
                  Tuple[chex.Array, chex.Array, List[float], bool],
                  Tuple[chex.Array, chex.Array, Dict[str, List[float]], bool]]

    :note: The concatenated parameters are split into control and delta. The training loop 
           tracks loss history, checks convergence every 100 steps, and supports early stopping.
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
        return theta_control_opt, delta_opt, history, converged
    elif return_history == "loss":
        return theta_control_opt, delta_opt, history["loss"], converged
    else:
        return theta_control_opt, delta_opt, converged


def multi_start_joint_train(
    n_restarts: int,
    init_theta_sampler: Callable[[], chex.Array],
    init_delta_sampler: Callable[[], chex.Array],
    agent: Callable,
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_treatment: List[Tuple[chex.Array, chex.Array]],
    n_params: int,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    delta_penalty_sigma: float = 1.0,
    min_num_converged: int = 3,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
    progress_bar: bool = True,
    get_history: bool = False,
) -> Union[Tuple[chex.Array, chex.Array, float], Tuple[chex.Array, chex.Array, float, chex.Array]]:
    """
    Run multiple joint training runs with different initializations and select the best model.

    Multiple runs are executed with random initializations for both control parameters and delta.
    For each run, the joint training function is called. The run with the best joint NLL is selected.
    Early stopping is triggered if a minimum number of runs have converged. Optionally, loss histories
    from all runs can be returned.

    :param n_restarts: Number of training runs.
    :type n_restarts: int
    :param init_theta_sampler: Function to generate a new control parameter array.
    :type init_theta_sampler: Callable[[], chex.Array]
    :param init_delta_sampler: Function to generate a new delta array.
    :type init_delta_sampler: Callable[[], chex.Array]
    :param agent: Joint model function.
    :type agent: Callable
    :param experiments_control: Dataset for the control group.
    :type experiments_control: List[Tuple[chex.Array, chex.Array]]
    :param experiments_treatment: Dataset for the treatment group.
    :type experiments_treatment: List[Tuple[chex.Array, chex.Array]]
    :param n_params: Number of elements in control parameters.
    :type n_params: int
    :param learning_rate: Optimizer learning rate.
    :type learning_rate: float
    :param num_steps: Maximum training steps per run.
    :type num_steps: int
    :param delta_penalty_sigma: Standard deviation for the penalty term.
    :type delta_penalty_sigma: float
    :param min_num_converged: Minimum converged runs required to trigger early stopping.
    :type min_num_converged: int
    :param verbose: If True, prints progress information.
    :type verbose: bool
    :param early_stopping: Early stopping parameters as a dict.
    :type early_stopping: Optional[Dict[str, float]]
    :param progress_bar: If True, shows a tqdm progress bar.
    :type progress_bar: bool
    :param get_history: If True, returns the loss histories from each run.
    :type get_history: bool

    :returns:
        - If get_history is False: (best_theta_control, best_delta, best_loss)
        - If get_history is True: ((best_theta_control, best_delta, best_loss), loss_history)
    :rtype: Union[Tuple[chex.Array, chex.Array, float], Tuple[Tuple[chex.Array, chex.Array, float], chex.Array]]

    :note: Each run is evaluated via total_nll_multi_group. Early stopping may occur
           if enough runs converge to the best loss.
    """
    best_theta = None
    best_delta = None
    best_loss = jnp.inf
    loss_history = []
    num_converged = 0

    for i in range(n_restarts):
        print(f"\n--- Joint Restart {i+1}/{n_restarts} ---")
        init_theta = init_theta_sampler()
        init_delta = init_delta_sampler()
        if get_history:
            theta_opt, delta_opt, history, converged = joint_train_model(
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
            loss_history.append(history)
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
        loss_history = jnp.array(loss_history)
        return best_theta, best_delta, best_loss, loss_history
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
    Evaluate the joint model by computing the NLL for the control and treatment groups,
    then summing them with the delta penalty to obtain the overall joint NLL.

    :param theta_control: Control group parameter array.
    :type theta_control: chex.Array
    :param delta: Delta offset parameter array.
    :type delta: chex.Array
    :param agent: Model function used to compute the NLL.
    :type agent: Callable
    :param experiments_control: Control group dataset (choices, rewards).
    :type experiments_control: List[Tuple[chex.Array, chex.Array]]
    :param experiments_treatment: Treatment group dataset.
    :type experiments_treatment: List[Tuple[chex.Array, chex.Array]]
    :param delta_penalty_sigma: Standard deviation for the penalty term.
    :type delta_penalty_sigma: float

    :returns: A tuple with (nll_control, nll_exp, joint_nll) where each value is a float.
    :rtype: Tuple[float, float, float]

    :note: The treatment group's NLL is computed using parameters
           theta_control + delta.
    """
    nll_control = total_negative_log_likelihood(theta_control, agent, experiments_control)
    theta_exp = theta_control + delta
    nll_exp = total_negative_log_likelihood(theta_exp, agent, experiments_treatment)
    penalty = jnp.sum((delta / delta_penalty_sigma) ** 2) / 2.0
    joint_nll = nll_control + nll_exp + penalty
    return float(nll_control), float(nll_exp), float(joint_nll)
