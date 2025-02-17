import jax
import jax.numpy as jnp
import chex
import optax
from typing import List, Tuple, Callable, Optional, Dict, Union
from tqdm.auto import trange
from flyjax.fitting.evaluation import total_negative_log_likelihood


def total_nll_hierarchical(
    theta_pop: chex.Array,  # Population-level (group-level) parameters, shape (n_params,)
    theta_subjects: chex.Array,  # Subject-specific parameters, shape (n_subjects, n_params)
    agent: Callable,
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    sigma_prior: float = 1.0,
) -> jnp.ndarray:
    """
    Compute the total negative log likelihood for a hierarchical model for one group.

    For each subject s:
      - Compute the NLL for that subject’s data using its parameter vector theta_subjects[s].
      - Add a quadratic penalty that encourages theta_subjects[s] to be close to theta_pop.

    Args:
        theta_pop: Population-level parameters (shape: [n_params]).
        theta_subjects: Subject-specific parameters (shape: [n_subjects, n_params]).
        agent: The agent model function that takes parameters and agent state and returns
            a tuple (action_probs, new_agent_state).
        experiments_by_subject: A list of length n_subjects; each element is a list of experiments
                                (each experiment is a tuple (choices, rewards)) for that subject.
        sigma_prior: Standard deviation for the Gaussian prior on the difference.

    Returns:
        The total negative log likelihood (sum over subjects).
    """
    total_nll = 0.0
    n_subjects = theta_subjects.shape[0]
    for s in range(n_subjects):
        theta_s = theta_subjects[s]
        subject_nll = 0.0
        # Sum NLL over all experiments for subject s.
        for exp in experiments_by_subject[s]:
            # Wrap the experiment tuple in a list.
            subject_nll += total_negative_log_likelihood(theta_s, agent, [exp])
        # Penalty term: (theta_s - theta_pop)^2/(2*sigma_prior^2)
        penalty = jnp.sum((theta_s - theta_pop) ** 2) / (2.0 * sigma_prior**2)
        total_nll += subject_nll + penalty
    return total_nll


def hierarchical_train_model(
    init_theta_pop: chex.Array,  # shape (n_params,)
    init_theta_subjects: chex.Array,  # shape (n_subjects, n_params)
    agent: Callable,
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    n_params: int = 4,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,  # increased from 1000
    sigma_prior: float = 1.0,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
) -> Tuple[chex.Array, chex.Array, bool]:
    """
    Jointly train the population-level parameters and the subject-specific parameters.

    The parameters are concatenated into a single vector.
    Convergence is checked every 100 steps using the relative change in loss.

    Returns:
        Optimized theta_pop (shape: [n_params]),
        Optimized theta_subjects (shape: [n_subjects, n_params]),
        and a converged flag indicating whether convergence was reached.
    """
    # Flatten subject parameters and concatenate with theta_pop.
    init_params = jnp.concatenate([init_theta_pop, init_theta_subjects.flatten()])
    optimizer = optax.adabelief(learning_rate)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def step_fn(params, opt_state):
        theta_pop = params[:n_params]
        # Reshape the remaining parameters into (n_subjects, n_params)
        theta_subjects = params[n_params:].reshape(-1, n_params)
        loss, grads = jax.value_and_grad(total_nll_hierarchical, argnums=(0, 1))(
            theta_pop, theta_subjects, agent, experiments_by_subject, sigma_prior
        )
        # Concatenate the gradients.
        grads_combined = jnp.concatenate([grads[0], grads[1].flatten()])
        updates, opt_state = optimizer.update(grads_combined, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, loss, opt_state

    params = init_params
    converged = False
    last_checkpoint_loss = None
    # Use the provided min_delta from early_stopping if available, otherwise default to 1e-2.
    convergence_threshold = (
        early_stopping.get("min_delta", 1e-2) if early_stopping is not None else 1e-2
    )
    iterator = trange(
        num_steps, desc="Hierarchical Training"
    )  # always using tqdm for clarity

    for i in iterator:
        params, loss, opt_state = step_fn(params, opt_state)
        loss_val = float(loss)

        # Every 100 steps, check convergence.
        if i % 100 == 0:
            if last_checkpoint_loss is not None:
                rel_change = abs(
                    (loss_val - last_checkpoint_loss) / last_checkpoint_loss
                )
                if rel_change < convergence_threshold:
                    if verbose:
                        print(
                            f"Convergence reached at step {i} with relative change {rel_change:.4f}."
                        )
                    converged = True
                    break
            last_checkpoint_loss = loss_val

            if verbose:
                print(f"Step {i:4d}, Hierarchical NLL: {loss_val:.4f}")

    theta_pop_opt = params[:n_params]
    theta_subjects_opt = params[n_params:].reshape(-1, n_params)
    return theta_pop_opt, theta_subjects_opt, converged


def multi_start_hierarchical_train(
    n_restarts: int,
    init_theta_pop_sampler: Callable[[], chex.Array],
    init_theta_subjects_sampler: Callable[[], chex.Array],
    agent: Callable,
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    n_params: int = 4,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,  # increased from 1000
    sigma_prior: float = 1.0,
    verbose: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
    min_num_converged: int = 3,
) -> Tuple[chex.Array, chex.Array, float]:
    """
    Run multiple training runs (with different random initializations) for the hierarchical model.

    Stops early if at least min_num_converged runs converge to the current best loss.

    Returns:
        Best theta_pop, best theta_subjects, and the best hierarchical loss.
    """
    best_theta_pop = None
    best_theta_subjects = None
    best_loss = jnp.inf
    all_losses = []
    num_converged = 0

    for i in range(n_restarts):
        print(f"\n--- Hierarchical Restart {i+1}/{n_restarts} ---")
        init_theta_pop = init_theta_pop_sampler()
        init_theta_subjects = init_theta_subjects_sampler()
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
        )
        hierarchical_nll = total_nll_hierarchical(
            theta_pop_opt,
            theta_subjects_opt,
            agent,
            experiments_by_subject,
            sigma_prior,
        )
        print(f"Restart {i+1} final Hierarchical NLL: {hierarchical_nll:.4f}")
        all_losses.append(float(hierarchical_nll))
        if converged:
            num_converged += 1
        if hierarchical_nll < best_loss:
            best_loss = hierarchical_nll
            best_theta_pop = theta_pop_opt
            best_theta_subjects = theta_subjects_opt
        if num_converged >= min_num_converged:
            print(
                f"Stopping early because {min_num_converged} runs have converged to the current best loss."
            )
            break

    print(f"\nBest Hierarchical NLL: {best_loss:.4f}")
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

    Returns:
        (nll_pop, nll_subjects, hierarchical_nll)
    """
    nll_pop = total_negative_log_likelihood(theta_pop, agent, experiments_by_subject[0])
    nll_subjects = 0.0
    for s, exps in enumerate(experiments_by_subject):
        nll_subjects += total_negative_log_likelihood(theta_subjects[s], agent, exps)
    penalty = jnp.sum((theta_subjects - theta_pop) ** 2) / (2.0 * sigma_prior**2)
    hierarchical_nll = nll_pop + nll_subjects + penalty
    return float(nll_pop), float(nll_subjects), float(hierarchical_nll)
