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
    Fit model parameters using AdaBelief optimization over simulated experiments.

    This function initializes the optimizer and iterates over a fixed number of training 
    steps. At each step, it computes the loss and performs a gradient update. The loss history 
    is recorded and checked every 100 iterations for convergence. Convergence is determined by the 
    relative change in loss, and an optional early stopping mechanism based on a patience counter 
    is also provided. A callback function may be executed at each iteration for custom logging.

    :param init_params: Initial parameter array.
    :type init_params: chex.Array
    :param agent: Function representing the agent which returns a tuple (action_probs, new_agent_state).
    :type agent: Callable
    :param experiments: List of experiments, each represented as a tuple (choices, rewards).
    :type experiments: List[Tuple[chex.Array, chex.Array]]
    :param learning_rate: Step size used by the AdaBelief optimizer (default: 5e-2).
    :type learning_rate: float
    :param num_steps: Maximum iterations for training (default: 10000).
    :type num_steps: int
    :param verbose: If True, prints loss information every 100 steps.
    :type verbose: bool
    :param progress_bar: If True, displays a progress bar via tqdm.
    :type progress_bar: bool
    :param callback: Optional function to be called at each iteration.
    :type callback: Optional[Callable[[int, chex.Array, float], None]]
    :param early_stopping: Dictionary specifying early stopping with keys "patience" and "min_delta".
    :type early_stopping: Optional[Dict[str, float]]
    :param return_history: Controls return values: "none" (only final params),
                           "loss" (list of losses), or "full" (dict with losses and parameters).
    :type return_history: str

    :returns: 
        - (params, converged_flag) if return_history is "none".
        - (params, loss_history, converged_flag) if return_history is "loss".
        - (params, history, converged_flag) if return_history is "full",
          where history is a dict containing keys "loss" and "params".
    :rtype: Union[Tuple[chex.Array, bool], Tuple[chex.Array, List[float], bool], Tuple[chex.Array, Dict[str, List[float]], bool]]

    :note:
        The training loop includes:
          - A JIT-compiled step function that computes gradients and updates parameters.
          - Loss history tracking for convergence analysis.
          - Convergence check every 100 steps based on relative loss change.
          - An optional early stopping mechanism if improvements are not significant.
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

    # Setup history tracking.
    history = {"loss": []}
    if return_history == "full":
        history["params"] = []  # Warning: can consume much memory.

    best_loss = jnp.inf
    best_params = init_params
    patience_counter = 0
    # Determine convergence threshold.
    convergence_threshold = (
        early_stopping.get("min_delta", 1e-2) if early_stopping is not None else 1e-2
    )

    last_checkpoint_loss = None
    converged = False

    # Select iterator: tqdm progress bar if enabled.
    iterator = trange(num_steps, desc="Training") if progress_bar else range(num_steps)

    params = init_params
    for i in iterator:
        params, opt_state, loss = step_fn(params, opt_state)
        loss_val = float(loss)

        # Record loss history.
        history["loss"].append(loss_val)
        if return_history == "full":
            history["params"].append(params)

        # Check convergence every 100 iterations.
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

        # Verbose logging.
        if verbose and (i % 100 == 0):
            print(f"Step {i:4d}, Negative Log Likelihood: {loss_val:.4f}")

        # Execute callback if provided.
        if callback is not None:
            callback(i, params, loss_val)

        # Implement patience-based early stopping if configured.
        if early_stopping is not None:
            if best_loss - loss_val > early_stopping.get("min_delta", 1e-4):
                best_loss = loss_val
                best_params = params
                patience_counter = 0  # Reset on improvement.
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
) -> Union[Tuple[chex.Array, float], Tuple[chex.Array, float, jnp.ndarray]]:
    """
    Execute multiple training runs with different random initializations and select the best model.

    This function runs the training process repeatedly using distinct random initializations.
    It monitors the negative log likelihood for each run and uses convergence criteria to determine 
    whether a run has sufficiently converged. If a minimum number of runs have converged to an 
    acceptable loss level, training is halted early. Optionally, the loss history across runs can be returned.

    :param n_restarts: Total number of training iterations with different initializations.
    :type n_restarts: int
    :param init_param_sampler: Function that generates a new initial parameter array.
    :type init_param_sampler: Callable[[], chex.Array]
    :param agent: Agent model function.
    :type agent: Callable
    :param training_experiments: List of experiments for training, each as (choices, rewards).
    :type training_experiments: List[Tuple[chex.Array, chex.Array]]
    :param learning_rate: Optimizer learning rate (default: 5e-2).
    :type learning_rate: float
    :param num_steps: Maximum training steps per run (default: 10000).
    :type num_steps: int
    :param min_num_converged: Minimum number of runs required to consider early stopping.
    :type min_num_converged: int
    :param verbose: If True, outputs progress information to the console.
    :type verbose: bool
    :param progress_bar: If True, displays a progress bar.
    :type progress_bar: bool
    :param early_stopping: Settings for early stopping with keys "patience" and "min_delta".
    :type early_stopping: Optional[Dict[str, float]]
    :param get_history: If True, return the loss history from each training run.
    :type get_history: bool

    :returns:
        - If get_history is False: tuple (best_params, best_loss).
        - If get_history is True: tuple (best_params, best_loss, loss_history),
          where loss_history is a jnp.ndarray recording losses from each run.
    :rtype: Union[Tuple[chex.Array, float], Tuple[chex.Array, float, jnp.ndarray]]

    :note:
        The function prints the negative log likelihood after each restart and halts early if the 
        criterion for the minimum number of converged runs is met.
    """
    best_params = None
    best_loss = jnp.inf
    loss_history = []
    num_converged = 0

    for i in range(n_restarts):
        print(f"\n--- Restart {i+1}/{n_restarts} ---")
        # Generate a new random initialization.
        init_params = init_param_sampler()
        if get_history:
            # Train the model and capture the loss history.
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
            loss_history.append(history)
        else:
            # Train the model without tracking loss history.
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
        train_nll = total_negative_log_likelihood(recovered_params, agent, training_experiments)
        print(f"Restart {i+1} final training NLL: {train_nll:.4f}")
        if converged:
            num_converged += 1
        if train_nll < best_loss:
            best_loss = train_nll
            best_params = recovered_params
        if num_converged >= min_num_converged:
            print(f"Stopping early because {min_num_converged} runs have converged to the current best loss.")
            break
        
    print(f"\nBest training NLL: {best_loss:.4f}")
    if get_history:
        # Convert loss_history to a jnp.ndarray.
        loss_history = jnp.array(loss_history)
        return best_params, best_loss, loss_history
    else:
        return best_params, best_loss


def evaluate_model(
    params: chex.Array,
    agent: Callable,
    experiments: List[Tuple[chex.Array, chex.Array]],
) -> float:
    """
    Assess model performance by computing the total negative log likelihood.

    This function runs the evaluation by forwarding the parameters, agent, 
    and list of experiments to the likelihood function. Lower negative log likelihood 
    values indicate a better fit.

    :param params: Model parameters.
    :type params: chex.Array
    :param agent: Agent model function.
    :type agent: Callable
    :param experiments: List of experiments, each as (choices, rewards).
    :type experiments: List[Tuple[chex.Array, chex.Array]]
    :returns: The computed negative log likelihood as a float.
    :rtype: float
    """
    nll = total_negative_log_likelihood(params, agent, experiments)
    return float(nll)
