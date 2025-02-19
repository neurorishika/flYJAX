import jax
import jax.numpy as jnp
import chex
import optax
from typing import List, Tuple, Callable, Optional, Dict, Union
from tqdm.auto import trange
from flyjax.fitting.evaluation import total_negative_log_likelihood


def train_model(
    init_params: chex.Array,
    agent: Callable,
    experiments: List[Tuple[chex.Array, chex.Array]],
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    verbose: bool = True,
    progress_bar: bool = True,
    callback: Optional[Callable[[int, chex.Array, float], None]] = None,
    early_stopping: Optional[Dict[str, float]] = None,
    return_history: str = "none",
) -> Union[Tuple[chex.Array, bool], Tuple[chex.Array, List[float], bool], Tuple[chex.Array, Dict[str, List[float]], bool]]:
    """
    Fit the model parameters to the simulated experiments using AdaBelief optimization.

    Additional features:
      - Convergence is checked every 100 steps: if the relative change in loss compared
        to 100 steps ago is less than the convergence threshold (default 1e-2), training stops.
      - The optimizer is AdaBelief with a learning rate of 5e-2.
      - Training runs for up to 10,000 steps unless early stopping criteria are met.

    Args:
        init_params: Initial guess for the parameters.
        agent: The agent model function that takes parameters and agent state and returns a tuple (action_probs, new_agent_state).
        experiments: List of experiments, each as a tuple (choices, rewards).
        learning_rate: Learning rate (default 5e-2).
        num_steps: Maximum number of training steps (default 10,000).
        verbose: If True, prints loss information every 100 steps.
        progress_bar: If True, uses a tqdm progress bar.
        callback: Optional function to call every iteration.
        early_stopping: Dictionary with keys "patience" and "min_delta" for additional early stopping.
        return_history: If "none", only return the final parameters. If "loss", return the history of losses. If "full", also return the history of parameters.

    Returns:
        If return_history is "none", returns a tuple (params, converged_flag).
        If return_history is "loss", returns a tuple (params, history, converged_flag) where history is a list of losses.
        If return_history is "full", returns a tuple (params, history, converged_flag) where history is a dictionary with keys "loss" and "params".
        where converged_flag indicates whether convergence (via the relative change test) was reached.
        In all cases, params is the final parameter array.
    """
    optimizer = optax.adabelief(learning_rate)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def step_fn(params, opt_state):
        loss, grads = jax.value_and_grad(total_negative_log_likelihood)(
            params, agent, experiments
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Setup history tracking
    history = {"loss": []}
    if return_history == "full":
        history["params"] = []  # be cautious: can grow large

    best_loss = jnp.inf
    best_params = init_params
    patience_counter = 0
    # Use early_stopping min_delta if provided, otherwise default convergence threshold 1e-2
    convergence_threshold = (
        early_stopping.get("min_delta", 1e-2) if early_stopping is not None else 1e-2
    )

    last_checkpoint_loss = None
    converged = False

    # Choose an iterator: tqdm if enabled, else simple range.
    iterator = trange(num_steps, desc="Training") if progress_bar else range(num_steps)

    params = init_params
    for i in iterator:
        params, opt_state, loss = step_fn(params, opt_state)
        loss_val = float(loss)

        # Update history.
        history["loss"].append(loss_val)
        if return_history == "full":
            history["params"].append(params)

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
                    best_params = params
                    converged = True
                    break
            last_checkpoint_loss = loss_val

        # Verbose printing every 100 steps.
        if verbose and (i % 100 == 0):
            print(f"Step {i:4d}, Negative Log Likelihood: {loss_val:.4f}")

        # Execute callback if provided.
        if callback is not None:
            callback(i, params, loss_val)

        # Additional early stopping (patience based) if provided.
        if early_stopping is not None:
            if best_loss - loss_val > early_stopping.get("min_delta", 1e-4):
                best_loss = loss_val
                best_params = params
                patience_counter = 0  # reset counter on improvement
            else:
                patience_counter += 1
                if patience_counter >= early_stopping.get("patience", 100):
                    if verbose:
                        print(f"Early stopping at step {i} with loss {loss_val:.4f}")
                    best_params = params
                    break

    if return_history == "full":
        return params, history, converged
    elif return_history == "loss":
        return params, history["loss"], converged
    else:
        return params, converged


def multi_start_train(
    n_restarts: int,
    init_param_sampler: Callable[[], chex.Array],
    agent: Callable,
    training_experiments: List[Tuple[chex.Array, chex.Array]],
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    min_num_converged: int = 3,
    verbose: bool = True,
    progress_bar: bool = True,
    early_stopping: Optional[Dict[str, float]] = None,
    get_history: bool = False,
) -> Union[Tuple[chex.Array, float], Tuple[chex.Array, float, Dict[int, List[float]]]]:
    """
    Perform multiple training runs with different random initializations and return the best parameters.

    In addition, the process stops early if at least 3 runs have converged to the current best.

    Args:
        n_restarts: Number of training runs.
        init_param_sampler: Function returning a new initial parameter array.
        agent: The agent model function.
        training_experiments: The dataset (list of experiments) to train on.
        learning_rate: Learning rate (default 5e-2).
        num_steps: Maximum training steps per run (default 10,000).
        min_num_converged: Minimum number of runs that must converge to the best loss.
        verbose: If True, prints progress information.
        progress_bar: If True, displays a tqdm progress bar.
        early_stopping: Dictionary with early stopping parameters.
        get_history: If True, return the history of losses for each run.

    Returns:
        A tuple (best_params, best_loss) where best_loss is the final negative log likelihood.
    """
    best_params = None
    best_loss = jnp.inf
    loss_history = {}
    num_converged = 0

    for i in range(n_restarts):
        print(f"\n--- Restart {i+1}/{n_restarts} ---")
        # Get a new random initialization.
        init_params = init_param_sampler()
        if get_history:
            # Train the model and return the loss history.
            recovered_params, history, converged = train_model(
                init_params,
                agent,
                training_experiments,
                learning_rate=learning_rate,
                num_steps=num_steps,
                verbose=verbose,
                progress_bar=progress_bar,
                early_stopping=early_stopping,
                return_history="loss",
            )
            loss_history[i] = history
        else:
            # Train the model
            recovered_params, converged = train_model(
                init_params,
                agent,
                training_experiments,
                learning_rate=learning_rate,
                num_steps=num_steps,
                verbose=verbose,
                progress_bar=progress_bar,
                early_stopping=early_stopping,
                return_history="none",
            )
        train_nll = total_negative_log_likelihood(
            recovered_params, agent, training_experiments
        )
        print(f"Restart {i+1} final training NLL: {train_nll:.4f}")
        if converged:
            num_converged += 1
        if train_nll < best_loss:
            best_loss = train_nll
            best_params = recovered_params
        if num_converged >= min_num_converged:
            print(
                f"Stopping early because {min_num_converged} runs have converged to the current best loss."
            )
            break

    print(f"\nBest training NLL: {best_loss:.4f}")
    if get_history:
        return best_params, best_loss, loss_history
    else:
        return best_params, best_loss


def evaluate_model(
    params: chex.Array,
    agent: Callable,
    experiments: List[Tuple[chex.Array, chex.Array]],
) -> float:
    """
    Evaluate the model on a set of experiments by computing the total negative log likelihood.

    Lower values indicate a better predictive fit.
    """
    nll = total_negative_log_likelihood(params, agent, experiments)
    return float(nll)
