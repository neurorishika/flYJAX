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
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    verbose: bool = True,
    progress_bar: bool = True,
    callback: Optional[Callable[[int, chex.Array, float], None]] = None,
    early_stopping: Optional[Dict[str, float]] = None,
    return_history: bool = False
) -> Union[chex.Array, Tuple[chex.Array, Dict[str, List]]]:
    """
    Fit the model parameters to the simulated experiments using Adam optimization.

    Additional features:
      - Verbose logging and optional tqdm progress bar.
      - An optional callback is called every iteration with the iteration index,
        current parameters, and loss.
      - Early stopping: If provided, training will stop early when the loss does not
        improve by at least `min_delta` for `patience` consecutive steps.
        Example: early_stopping = {"patience": 100, "min_delta": 1e-4}
      - History tracking: If return_history=True, returns a dictionary containing the loss
        history and (optionally) parameter history.

    Args:
        init_params: Initial guess for the parameters.
        agent: The agent model function that takes parameters and agent state and returns
            a tuple (action_probs, new_agent_state).
        experiments: List of experiments, each as a tuple (choices, rewards).
        learning_rate: Learning rate for the Adam optimizer.
        num_steps: Maximum number of training steps.
        verbose: If True, prints loss information at regular intervals.
        progress_bar: If True, uses a tqdm progress bar.
        callback: Optional callable f(step, params, loss) executed each iteration.
        early_stopping: Optional dictionary with keys "patience" (int) and "min_delta" (float).
        return_history: If True, returns a history dictionary along with the final parameters.

    Returns:
        If return_history is False: final optimized parameters.
        If return_history is True: a tuple (best_params, history_dict), where history_dict contains:
           - "loss": List of loss values per step.
           - "params": (Optional) List of parameter values (can be large).
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)
    
    @jax.jit
    def step_fn(params, opt_state):
        loss, grads = jax.value_and_grad(total_negative_log_likelihood)(params, agent, experiments)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Setup history tracking
    history = {"loss": []}
    if return_history:
        history["params"] = []  # be cautious: this can grow large for many steps

    best_loss = jnp.inf
    best_params = init_params
    patience_counter = 0
    patience = early_stopping.get("patience", 100) if early_stopping is not None else None
    min_delta = early_stopping.get("min_delta", 1e-4) if early_stopping is not None else None

    # Choose an iterator: either a tqdm progress bar or a simple range.
    iterator = trange(num_steps, desc="Training") if progress_bar else range(num_steps)

    params = init_params
    for i in iterator:
        params, opt_state, loss = step_fn(params, opt_state)
        loss_val = float(loss)

        # Update history.
        history["loss"].append(loss_val)
        if return_history:
            history["params"].append(params)

        # Verbose printing at regular intervals.
        if verbose and (i % 100 == 0):
            print(f"Step {i:4d}, Negative Log Likelihood: {loss_val:.4f}")

        # Execute callback if provided.
        if callback is not None:
            callback(i, params, loss_val)

        # Early stopping check.
        if early_stopping is not None:
            if best_loss - loss_val > min_delta:
                best_loss = loss_val
                best_params = params
                patience_counter = 0  # reset the counter on improvement
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at step {i} with loss {loss_val:.4f}")
                    params = best_params
                    break

    if return_history:
        return params, history
    return params


def multi_start_train(
    n_restarts: int,
    init_param_sampler: Callable[[], chex.Array],
    agent: Callable,
    training_experiments: List[Tuple[chex.Array, chex.Array]],
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    verbose: bool = True,
    progress_bar: bool = True,
    early_stopping: Optional[Dict[str, float]] = None
) -> Tuple[chex.Array, List[float]]:
    """
    Perform multiple training runs (with different random initializations) and return the best parameters.
    
    Args:
        n_restarts: Number of independent training runs.
        init_param_sampler: A function that returns an initial parameter array.
        agent: The agent model function that takes parameters and agent state and returns
        training_experiments: The dataset (list of experiments) to train on.
        learning_rate: Learning rate for training.
        num_steps: Number of training steps per run.
        verbose: If True, prints information during training.
        progress_bar: If True, displays a progress bar.
        early_stopping: Early stopping options as in train_model.
        
    Returns:
        best_params: The recovered parameters with the lowest training negative log likelihood.
        losses: A list of final training losses for each restart.
    """
    best_params = None
    best_loss = jnp.inf
    all_losses = []
    
    for i in range(n_restarts):
        print(f"\n--- Restart {i+1}/{n_restarts} ---")
        # Get a new random initialization.
        init_params = init_param_sampler()
        # Train the model.
        recovered_params = train_model(
            init_params, 
            agent,
            training_experiments,
            learning_rate=learning_rate,
            num_steps=num_steps,
            verbose=verbose,
            progress_bar=progress_bar,
            early_stopping=early_stopping
        )
        # Evaluate training performance (e.g., final negative log likelihood).
        train_nll = total_negative_log_likelihood(recovered_params, agent, training_experiments)
        print(f"Restart {i+1} final training NLL: {train_nll:.4f}")
        all_losses.append(float(train_nll))
        if train_nll < best_loss:
            best_loss = train_nll
            best_params = recovered_params

    print(f"\nBest training NLL: {best_loss:.4f}")
    return best_params, best_loss

def evaluate_model(
    params: chex.Array,
    agent: Callable,
    experiments: List[Tuple[chex.Array, chex.Array]]
) -> float:
    """
    Evaluate the model on a set of experiments by computing the total negative log likelihood.
    
    Lower values indicate a better predictive fit.
    """
    nll = total_negative_log_likelihood(params, agent, experiments)
    return float(nll)
