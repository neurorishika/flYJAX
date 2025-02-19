# flyjax/fitting/hierarchical.py
import jax
import jax.numpy as jnp
import chex
import optax
from typing import List, Tuple, Callable, Optional, Dict, Union
from tqdm.auto import trange
from flyjax.fitting.evaluation import total_negative_log_likelihood

def total_nll_hierarchical(
    theta_pop: chex.Array,  # Population-level parameters, shape (n_params,)
    theta_subjects: chex.Array,  # Subject-specific parameters, shape (n_subjects, n_params)
    agent: Callable,
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    sigma_prior: float = 1.0,
) -> jnp.ndarray:
    """
    Compute the total negative log likelihood for a hierarchical model.

    For each subject s:
      - Compute the NLL for that subject’s data using theta_subjects[s].
      - Add a quadratic penalty that encourages theta_subjects[s] to be close to theta_pop.

    Args:
        theta_pop: Population-level parameters (shape: [n_params]).
        theta_subjects: Subject-specific parameters (shape: [n_subjects, n_params]).
        agent: The agent model function (signature: (params, agent_state, choice, reward)).
        experiments_by_subject: A list (length = n_subjects) where each element is a list of experiments,
                                each experiment being a tuple (choices, rewards).
        sigma_prior: Standard deviation for the Gaussian prior.

    Returns:
        The total negative log likelihood (sum over subjects).
    """
    total_nll = 0.0
    n_subjects = theta_subjects.shape[0]
    for s in range(n_subjects):
        theta_s = theta_subjects[s]
        subject_nll = 0.0
        # Sum NLL over all experiments for this subject.
        for exp in experiments_by_subject[s]:
            subject_nll += total_negative_log_likelihood(theta_s, agent, [exp])
        penalty = jnp.sum((theta_s - theta_pop) ** 2) / (2.0 * sigma_prior**2)
        total_nll += subject_nll + penalty
    return total_nll


def hierarchical_train_model(
    init_theta_pop: chex.Array,
    init_theta_subjects: chex.Array,
    agent: Callable,
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    n_params: int = 4,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    sigma_prior: float = 1.0,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
    progress_bar: bool = True,
    return_history: str = "none",  # Options: "none", "loss", "full"
    callback: Optional[Callable[[int, chex.Array, float], None]] = None,
) -> Union[
    Tuple[chex.Array, chex.Array, bool],
    Tuple[Tuple[chex.Array, chex.Array, bool], Dict[str, List]]
]:
    """
    Jointly train the population-level and subject-specific parameters.

    The parameters are concatenated into a single vector. Convergence is checked every 100 steps
    (based on relative change in loss). Optionally tracks history and calls a callback each iteration.

    Args:
        init_theta_pop: Initial guess for population parameters (shape: [n_params]).
        init_theta_subjects: Initial guess for subject parameters (shape: [n_subjects, n_params]).
        agent: The agent model function.
        experiments_by_subject: List of subject experiment lists.
        n_params: Number of parameters in theta_pop.
        learning_rate: Learning rate for AdaBelief.
        num_steps: Maximum training steps.
        sigma_prior: Std. dev. for the quadratic penalty.
        verbose: If True, prints progress every 100 steps.
        early_stopping: Dictionary with keys "min_delta" and "patience" for early stopping.
        progress_bar: If True, shows a progress bar.
        return_history: "none" returns only final parameters and convergence flag;
                        "loss" returns a list of losses;
                        "full" returns a dictionary with keys "loss" and "params".
        callback: Optional callback function called every iteration with (step, params, loss).

    Returns:
        If return_history == "none":
            (theta_pop_opt, theta_subjects_opt, converged)
        If "loss":
            ((theta_pop_opt, theta_subjects_opt, converged), loss_history)
        If "full":
            ((theta_pop_opt, theta_subjects_opt, converged), full_history)
    """
    # Concatenate population and subject parameters.
    init_params = jnp.concatenate([init_theta_pop, init_theta_subjects.flatten()])
    optimizer = optax.adabelief(learning_rate)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def step_fn(params, opt_state):
        theta_pop = params[:n_params]
        theta_subjects = params[n_params:].reshape(-1, n_params)
        loss, grads = jax.value_and_grad(total_nll_hierarchical, argnums=(0, 1))(
            theta_pop, theta_subjects, agent, experiments_by_subject, sigma_prior
        )
        grads_combined = jnp.concatenate([grads[0], grads[1].flatten()])
        updates, opt_state = optimizer.update(grads_combined, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, loss, opt_state

    # History tracking
    history = {"loss": []}
    if return_history == "full":
        history["params"] = []

    last_checkpoint_loss = None
    converged = False
    patience_counter = 0
    best_loss = float('inf')
    # Set convergence threshold.
    convergence_threshold = early_stopping.get("min_delta", 1e-2) if early_stopping else 1e-2

    iterator = trange(num_steps, desc="Hierarchical Training") if progress_bar else range(num_steps)
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
                    converged = True
                    break
            last_checkpoint_loss = loss_val
            if verbose:
                print(f"Step {i:4d}, Hierarchical NLL: {loss_val:.4f}")
        if callback is not None:
            callback(i, params, loss_val)
        if early_stopping is not None:
            if i == 0:
                best_loss = loss_val
            else:
                if best_loss - loss_val > early_stopping.get("min_delta", 1e-4):
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping.get("patience", 100):
                        if verbose:
                            print(f"Early stopping at step {i} with loss {loss_val:.4f}")
                        break

    theta_pop_opt = params[:n_params]
    theta_subjects_opt = params[n_params:].reshape(-1, n_params)
    if return_history == "full":
        return (theta_pop_opt, theta_subjects_opt, converged), history
    elif return_history == "loss":
        return (theta_pop_opt, theta_subjects_opt, converged), history["loss"]
    else:
        return theta_pop_opt, theta_subjects_opt, converged


def multi_start_hierarchical_train(
    n_restarts: int,
    init_theta_pop_sampler: Callable[[], chex.Array],
    init_theta_subjects_sampler: Callable[[], chex.Array],
    agent: Callable,
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    n_params: int = 4,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    sigma_prior: float = 1.0,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
    min_num_converged: int = 3,
    progress_bar: bool = True,
    get_history: bool = False,
) -> Union[
    Tuple[chex.Array, chex.Array, float],
    Tuple[Tuple[chex.Array, chex.Array, float], Dict[int, List]]
]:
    """
    Run multiple training runs (with different random initializations) for the hierarchical model.
    Stops early if at least min_num_converged runs have converged to the best loss.

    Args:
        n_restarts: Number of training runs.
        init_theta_pop_sampler: Function to generate a new initial population parameter vector.
        init_theta_subjects_sampler: Function to generate a new initial subject parameters array.
        agent: The agent model function.
        experiments_by_subject: List of subject experiment lists.
        n_params: Number of parameters in theta_pop.
        learning_rate: Learning rate.
        num_steps: Maximum training steps per run.
        sigma_prior: Standard deviation for the penalty.
        verbose: If True, prints progress.
        early_stopping: Dictionary with early stopping parameters.
        min_num_converged: Minimum number of converged runs before stopping.
        progress_bar: If True, shows a progress bar.
        get_history: If True, returns the loss history for each run.

    Returns:
        If get_history is False:
            (best_theta_pop, best_theta_subjects, best_loss)
        If get_history is True:
            ((best_theta_pop, best_theta_subjects, best_loss), loss_history)
    """
    best_theta_pop = None
    best_theta_subjects = None
    best_loss = float('inf')
    loss_history = {}
    num_converged = 0

    for i in range(n_restarts):
        print(f"\n--- Hierarchical Restart {i+1}/{n_restarts} ---")
        init_theta_pop = init_theta_pop_sampler()
        init_theta_subjects = init_theta_subjects_sampler()
        if get_history:
            (theta_pop_opt, theta_subjects_opt, converged), run_history = hierarchical_train_model(
                init_theta_pop,
                init_theta_subjects,
                agent,
                experiments_by_subject,
                n_params=n_params,
                learning_rate=learning_rate,
                num_steps=num_steps,
                sigma_prior=sigma_prior,
                verbose=verbose,
                early_stopping=early_stopping,
                progress_bar=progress_bar,
                return_history="loss",
            )
            loss_history[i] = run_history
        else:
            theta_pop_opt, theta_subjects_opt, converged = hierarchical_train_model(
                init_theta_pop,
                init_theta_subjects,
                agent,
                experiments_by_subject,
                n_params=n_params,
                learning_rate=learning_rate,
                num_steps=num_steps,
                sigma_prior=sigma_prior,
                verbose=verbose,
                early_stopping=early_stopping,
                progress_bar=progress_bar,
                return_history="none",
            )
        current_loss = total_nll_hierarchical(theta_pop_opt, theta_subjects_opt, agent, experiments_by_subject, sigma_prior)
        print(f"Restart {i+1} final Hierarchical NLL: {current_loss:.4f}")
        if converged:
            num_converged += 1
        if current_loss < best_loss:
            best_loss = current_loss
            best_theta_pop = theta_pop_opt
            best_theta_subjects = theta_subjects_opt
        if num_converged >= min_num_converged:
            print(f"Stopping early because {min_num_converged} runs have converged to the current best loss.")
            break

    print(f"\nBest Hierarchical NLL: {best_loss:.4f}")
    if get_history:
        return (best_theta_pop, best_theta_subjects, best_loss), loss_history
    else:
        return best_theta_pop, best_theta_subjects, best_loss


def evaluate_hierarchical_model(
    theta_pop: chex.Array,
    theta_subjects: chex.Array,
    agent: Callable,
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    sigma_prior: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Evaluate the hierarchical model by computing NLLs for the population-level data,
    subject-specific data, and the total hierarchical NLL.

    Args:
        theta_pop: Population-level parameters.
        theta_subjects: Subject-specific parameters.
        agent: The agent model function.
        experiments_by_subject: List of subject experiment lists.
        sigma_prior: Std. dev. for the penalty.

    Returns:
        A tuple (nll_pop, nll_subjects, hierarchical_nll).
    """
    nll_pop = total_negative_log_likelihood(theta_pop, agent, experiments_by_subject[0])
    nll_subjects = 0.0
    for s, exps in enumerate(experiments_by_subject):
        nll_subjects += total_negative_log_likelihood(theta_subjects[s], agent, exps)
    penalty = jnp.sum((theta_subjects - theta_pop) ** 2) / (2.0 * sigma_prior**2)
    hierarchical_nll = nll_pop + nll_subjects + penalty
    return float(nll_pop), float(nll_subjects), float(hierarchical_nll)