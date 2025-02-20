import jax
import jax.numpy as jnp
import chex
import optax
from typing import List, Tuple, Callable, Optional, Dict, Union
from tqdm.auto import trange
from flyjax.fitting.evaluation import total_negative_log_likelihood

def total_nll_hierarchical(
    theta_pop: chex.Array,  # Population-level parameters (n_params,)
    theta_subjects: chex.Array,  # Subject-specific parameters (n_subjects, n_params)
    agent: Callable,
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    sigma_prior: float = 1.0,
) -> jnp.ndarray:
    """
    Compute the total negative log likelihood (NLL) for the hierarchical model.

    For each subject:
      - Evaluate the model NLL using the subject-specific parameters.
      - Add a quadratic penalty that encourages subject parameters to remain near the population mean.

    :param theta_pop: Population-level parameter vector.
    :type theta_pop: chex.Array
    :param theta_subjects: Subject-specific parameter matrix.
    :type theta_subjects: chex.Array
    :param agent: Agent model function for likelihood evaluation.
    :type agent: Callable
    :param experiments_by_subject: List (per subject) of experiments; each experiment is a tuple (choices, rewards).
    :type experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]]
    :param sigma_prior: Standard deviation for the Gaussian prior penalty.
    :type sigma_prior: float

    :returns: The total hierarchical NLL (sum over subjects and penalty).
    :rtype: jnp.ndarray

    :note: This function iterates over subjects, summing each subject’s NLL (across experiments)
           and adding a regularization term based on the deviation from theta_pop.
    """
    total_nll = 0.0
    n_subjects = theta_subjects.shape[0]
    for s in range(n_subjects):
        theta_s = theta_subjects[s]
        subject_nll = 0.0
        # Sum the NLL for all experiments of this subject.
        for exp in experiments_by_subject[s]:
            subject_nll += total_negative_log_likelihood(theta_s, agent, [exp])
        # Apply a quadratic penalty to encourage theta_s to stay close to theta_pop.
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
) -> Union[Tuple[chex.Array, chex.Array, bool],
           Tuple[chex.Array, chex.Array, List[float], bool],
           Tuple[chex.Array, chex.Array, Dict[str, List[float]], bool]]:
    """
    Jointly train the population-level and subject-level parameters using AdaBelief.

    This function concatenates the population and subject parameters into a single vector, then
    iteratively updates them via gradient descent. The total hierarchical NLL (data fit plus prior penalty)
    is minimized. Loss history is recorded and convergence is checked every 100 iterations, with options
    for early stopping and custom callback execution.

    :param init_theta_pop: Initial guess for the population parameters.
    :type init_theta_pop: chex.Array
    :param init_theta_subjects: Initial guess for the subject-specific parameters.
    :type init_theta_subjects: chex.Array
    :param agent: Agent model function used for computing likelihood.
    :type agent: Callable
    :param experiments_by_subject: List of experiment lists per subject.
    :type experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]]
    :param n_params: Number of population-level parameters.
    :type n_params: int
    :param learning_rate: Optimizer learning rate.
    :type learning_rate: float
    :param num_steps: Maximum number of training iterations.
    :type num_steps: int
    :param sigma_prior: Standard deviation used in the quadratic prior penalty.
    :type sigma_prior: float
    :param verbose: If True, prints progress information every 100 steps.
    :type verbose: bool
    :param early_stopping: Dictionary with keys "min_delta" and "patience" for early stopping.
    :type early_stopping: Optional[Dict[str, float]]
    :param progress_bar: If True, displays a tqdm progress bar.
    :type progress_bar: bool
    :param return_history: Return format ("none", "loss", or "full").
    :type return_history: str
    :param callback: Optional callback to call at each training iteration.
    :type callback: Optional[Callable[[int, chex.Array, float], None]]

    :returns:
        - If "none": (theta_pop_opt, theta_subjects_opt, converged_flag)
        - If "loss": (theta_pop_opt, theta_subjects_opt, loss_history, converged_flag)
        - If "full": (theta_pop_opt, theta_subjects_opt, full_history, converged_flag)
    :rtype: Union[Tuple[chex.Array, chex.Array, bool],
                  Tuple[chex.Array, chex.Array, List[float], bool],
                  Tuple[chex.Array, chex.Array, Dict[str, List[float]], bool]]

    :note: After each iteration, the concatenated parameter vector is updated. The population
           parameters are the first n_params elements; the remaining elements are reshaped for subjects.
    """
    # Concatenate population and flattened subject parameters.
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

    # Setup history tracking.
    history = {"loss": []}
    if return_history == "full":
        history["params"] = []

    last_checkpoint_loss = None
    converged = False
    patience_counter = 0
    best_loss = float('inf')
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
        return theta_pop_opt, theta_subjects_opt, history, converged
    elif return_history == "loss":
        return theta_pop_opt, theta_subjects_opt, history["loss"], converged
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
    Tuple[chex.Array, chex.Array, float, chex.Array]
]:
    """
    Run multiple training runs for the hierarchical model with random initializations,
    and select the best set of parameters.

    Each run invokes the hierarchical_train_model function. The run with the lowest hierarchical NLL
    is chosen, and early stopping may be triggered if a minimum number of runs have converged.
    Optionally, loss histories for all runs can also be returned.

    :param n_restarts: Number of training runs.
    :type n_restarts: int
    :param init_theta_pop_sampler: Function to generate a new population parameter vector.
    :type init_theta_pop_sampler: Callable[[], chex.Array]
    :param init_theta_subjects_sampler: Function to generate a new subject parameter array.
    :type init_theta_subjects_sampler: Callable[[], chex.Array]
    :param agent: Agent model function.
    :type agent: Callable
    :param experiments_by_subject: List of experiment lists per subject.
    :type experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]]
    :param n_params: Number of elements in population parameters.
    :type n_params: int
    :param learning_rate: Optimizer learning rate.
    :type learning_rate: float
    :param num_steps: Maximum training steps per run.
    :type num_steps: int
    :param sigma_prior: Std. dev. for the prior penalty.
    :type sigma_prior: float
    :param verbose: If True, prints progress during training.
    :type verbose: bool
    :param early_stopping: Early stopping parameters as a dict.
    :type early_stopping: Optional[Dict[str, float]]
    :param min_num_converged: Minimum number of runs required to stop early.
    :type min_num_converged: int
    :param progress_bar: If True, displays a progress bar.
    :type progress_bar: bool
    :param get_history: If True, returns loss history for each run.
    :type get_history: bool

    :returns:
        - If get_history is False: (best_theta_pop, best_theta_subjects, best_loss)
        - If get_history is True: ((best_theta_pop, best_theta_subjects, best_loss), loss_history)
    :rtype: Union[Tuple[chex.Array, chex.Array, float], Tuple[Tuple[chex.Array, chex.Array, float], chex.Array]]

    :note: Loss is assessed using total_nll_hierarchical.
    """
    best_theta_pop = None
    best_theta_subjects = None
    best_loss = float('inf')
    loss_history = []
    num_converged = 0

    for i in range(n_restarts):
        print(f"\n--- Hierarchical Restart {i+1}/{n_restarts} ---")
        init_theta_pop = init_theta_pop_sampler()
        init_theta_subjects = init_theta_subjects_sampler()
        if get_history:
            theta_pop_opt, theta_subjects_opt, history, converged = hierarchical_train_model(
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
            loss_history.append(history)
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
        loss_history = jnp.array(loss_history)
        return best_theta_pop, best_theta_subjects, best_loss, loss_history
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
    Evaluate the hierarchical model by computing separate NLLs for the population-level and 
    subject-specific data, then summing these along with the prior penalty.

    :param theta_pop: Population-level parameters.
    :type theta_pop: chex.Array
    :param theta_subjects: Subject-specific parameters.
    :type theta_subjects: chex.Array
    :param agent: Agent model function for likelihood evaluation.
    :type agent: Callable
    :param experiments_by_subject: List of experiments per subject.
    :type experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]]
    :param sigma_prior: Std. dev. for the quadratic penalty.
    :type sigma_prior: float

    :returns: A tuple (nll_pop, nll_subjects, hierarchical_nll) each as a float.
    :rtype: Tuple[float, float, float]

    :note: The overall hierarchical NLL is computed as the sum of the population NLL, subject NLL,
           and the regularization penalty.
    """
    nll_pop = total_negative_log_likelihood(theta_pop, agent, experiments_by_subject[0])
    nll_subjects = 0.0
    for s, exps in enumerate(experiments_by_subject):
        nll_subjects += total_negative_log_likelihood(theta_subjects[s], agent, exps)
    penalty = jnp.sum((theta_subjects - theta_pop) ** 2) / (2.0 * sigma_prior**2)
    hierarchical_nll = nll_pop + nll_subjects + penalty
    return float(nll_pop), float(nll_subjects), float(hierarchical_nll)