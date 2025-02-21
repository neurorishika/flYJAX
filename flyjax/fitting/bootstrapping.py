"""
flyjax/fitting/bootstrapping.py

This module implements bootstrapping routines for model fitting. The
functions provided here allow you to compute bootstrapped estimates for
single-group, joint, or hierarchical models. Bootstrapping is done by
resampling the training data with replacement, and then running the model
training (using the fixed training schedules from multi-start functions)
on each resampled dataset. Optional process-based parallelization and chunking
are provided to accelerate the procedure.
"""

import jax
import jax.numpy as jnp
import chex
import optax
import numpy as np
from functools import partial
from typing import List, Tuple, Callable, Optional, Dict, Union
from concurrent.futures import ProcessPoolExecutor

# Import existing training functions from the codebase.
from flyjax.fitting.train import multi_start_train
from flyjax.fitting.joint import multi_start_joint_train
from flyjax.fitting.hierarchical import multi_start_hierarchical_train

# Import evaluation (if desired) for reporting losses.
from flyjax.fitting.evaluation import total_negative_log_likelihood

###############################################################################
# Helper functions (module-level)
###############################################################################

def _resample_experiments(
    experiments: Tuple[Tuple[chex.Array, chex.Array], ...],
    key: chex.Array
) -> List[Tuple[chex.Array, chex.Array]]:
    """
    Resample a list of experiments with replacement.
    """
    n = len(experiments)
    # Use a concrete numpy array for the candidate indices.
    a = np.arange(n)
    indices = jax.random.choice(key, a=a, shape=(n,), replace=True)
    # Force the indices to be concrete.
    indices = jax.device_get(indices)
    return [experiments[i] for i in indices.tolist()]

def _resample_experiments_joint(
    experiments: Tuple[Tuple[chex.Array, chex.Array], ...],
    key: chex.Array
) -> List[Tuple[chex.Array, chex.Array]]:
    """
    Resample joint experiments (for a given group) with replacement.
    """
    return _resample_experiments(experiments, key)

def _resample_by_subject(
    experiments_by_subject: Tuple[Tuple[Tuple[chex.Array, chex.Array], ...], ...],
    key: chex.Array
) -> List[List[Tuple[chex.Array, chex.Array]]]:
    """
    Resample a list of subjects (each subject is a list of experiments)
    with replacement.
    """
    n = len(experiments_by_subject)
    a = np.arange(n)
    indices = jax.random.choice(key, a=a, shape=(n,), replace=True)
    indices = jax.device_get(indices)
    return [list(experiments_by_subject[i]) for i in indices.tolist()]

###############################################################################
# Top-level worker functions for parallel execution
###############################################################################

def _single_bootstrap_run(
    seed: int,
    experiments_static: Tuple[Tuple[chex.Array, chex.Array], ...],
    n_restarts: int,
    init_param_sampler: Callable[[], chex.Array],
    agent: Callable,
    learning_rate: float,
    num_steps: int,
    min_num_converged: int,
    early_stopping: Optional[Dict[str, float]],
    get_history: bool
) -> Union[chex.Array, Tuple[chex.Array, jnp.ndarray]]:
    key = jax.random.PRNGKey(seed)
    boot_experiments = _resample_experiments(experiments_static, key)
    if get_history:
        params, loss_history, _ = multi_start_train(
            n_restarts=n_restarts,
            init_param_sampler=init_param_sampler,
            agent=agent,
            training_experiments=boot_experiments,
            learning_rate=learning_rate,
            num_steps=num_steps,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            get_history=True,
        )
        return params, loss_history
    else:
        params, _ = multi_start_train(
            n_restarts=n_restarts,
            init_param_sampler=init_param_sampler,
            agent=agent,
            training_experiments=boot_experiments,
            learning_rate=learning_rate,
            num_steps=num_steps,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            get_history=False,
        )
        return params

def _joint_bootstrap_run(
    seed: int,
    control_static: Tuple[Tuple[chex.Array, chex.Array], ...],
    treatment_static: Tuple[Tuple[chex.Array, chex.Array], ...],
    n_params: int,
    init_theta_sampler: Callable[[], chex.Array],
    init_delta_sampler: Callable[[], chex.Array],
    agent: Callable,
    learning_rate: float,
    num_steps: int,
    n_restarts: int,
    min_num_converged: int,
    early_stopping: Optional[Dict[str, float]],
    delta_penalty_sigma: float,
    get_history: bool
) -> Tuple[chex.Array, chex.Array]:
    key = jax.random.PRNGKey(seed)
    boot_control = _resample_experiments_joint(control_static, key)
    key, subkey = jax.random.split(key)
    boot_treatment = _resample_experiments_joint(treatment_static, subkey)
    if get_history:
        theta, delta, _, _ = multi_start_joint_train(
            init_theta_sampler=init_theta_sampler,
            init_delta_sampler=init_delta_sampler,
            agent=agent,
            n_params=n_params,
            experiments_control=boot_control,
            experiments_treatment=boot_treatment,
            learning_rate=learning_rate,
            num_steps=num_steps,
            n_restarts=n_restarts,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            delta_penalty_sigma=delta_penalty_sigma,
            get_history=True,
        )
    else:
        theta, delta, _ = multi_start_joint_train(
            init_theta_sampler=init_theta_sampler,
            init_delta_sampler=init_delta_sampler,
            agent=agent,
            n_params=n_params,
            experiments_control=boot_control,
            experiments_treatment=boot_treatment,
            learning_rate=learning_rate,
            num_steps=num_steps,
            n_restarts=n_restarts,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            delta_penalty_sigma=delta_penalty_sigma,
            get_history=False,
        )
    return theta, delta

def _hierarchical_bootstrap_run(
    seed: int,
    subjects_static: Tuple[Tuple[Tuple[chex.Array, chex.Array], ...], ...],
    n_params: int,
    init_theta_pop_sampler: Callable[[], chex.Array],
    make_sample_init_theta_subjects: Callable[[int], Callable[[], chex.Array]],
    agent: Callable,
    sigma_prior: float,
    learning_rate: float,
    num_steps: int,
    n_restarts: int,
    min_num_converged: int,
    early_stopping: Optional[Dict[str, float]],
    get_history: bool
) -> Tuple[chex.Array, chex.Array]:
    key = jax.random.PRNGKey(seed)
    boot_subjects = _resample_by_subject(subjects_static, key)
    n_train_subjects = len(boot_subjects)
    local_sampler = make_sample_init_theta_subjects(n_train_subjects)
    if get_history:
        theta_pop, theta_subjects, _, _ = multi_start_hierarchical_train(
            n_params=n_params,
            n_restarts=n_restarts,
            init_theta_pop_sampler=init_theta_pop_sampler,
            init_theta_subjects_sampler=local_sampler,
            agent=agent,
            experiments_by_subject=boot_subjects,
            learning_rate=learning_rate,
            num_steps=num_steps,
            sigma_prior=sigma_prior,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            get_history=True,
        )
    else:
        theta_pop, theta_subjects, _ = multi_start_hierarchical_train(
            n_params=n_params,
            n_restarts=n_restarts,
            init_theta_pop_sampler=init_theta_pop_sampler,
            init_theta_subjects_sampler=local_sampler,
            agent=agent,
            experiments_by_subject=boot_subjects,
            learning_rate=learning_rate,
            num_steps=num_steps,
            sigma_prior=sigma_prior,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            get_history=False,
        )
    return theta_pop, theta_subjects

###############################################################################
# Bootstrapping functions for Single-Group Models
###############################################################################

def bootstrap_train_single(
    training_experiments: List[Tuple[chex.Array, chex.Array]],
    n_bootstrap: int,
    init_param_sampler: Callable[[], chex.Array],
    agent: Callable,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
    use_parallel: bool = True,
    get_history: bool = False
) -> Union[
    Dict[int, chex.Array],
    Dict[int, Tuple[chex.Array, jnp.ndarray]]
]:
    """
    Run bootstrap replications for a single-group model training using process-based parallelization.
    """
    experiments_static = tuple(training_experiments)
    seeds = list(range(n_bootstrap))
    if use_parallel:
        with ProcessPoolExecutor() as executor:
            results = {seed: result for seed, result in zip(
                seeds,
                executor.map(
                    _single_bootstrap_run,
                    seeds,
                    [experiments_static] * n_bootstrap,
                    [n_restarts] * n_bootstrap,
                    [init_param_sampler] * n_bootstrap,
                    [agent] * n_bootstrap,
                    [learning_rate] * n_bootstrap,
                    [num_steps] * n_bootstrap,
                    [min_num_converged] * n_bootstrap,
                    [early_stopping] * n_bootstrap,
                    [get_history] * n_bootstrap,
                )
            )}
        return results
    else:
        results = {}
        for seed in seeds:
            results[seed] = _single_bootstrap_run(
                seed, experiments_static, n_restarts, init_param_sampler, agent,
                learning_rate, num_steps, min_num_converged, early_stopping, get_history
            )
        return results

###############################################################################
# Bootstrapping functions for Joint Models
###############################################################################

def bootstrap_train_joint(
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_treatment: List[Tuple[chex.Array, chex.Array]],
    n_bootstrap: int,
    n_params: int,
    init_theta_sampler: Callable[[], chex.Array],
    init_delta_sampler: Callable[[], chex.Array],
    agent: Callable,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
    delta_penalty_sigma: float = 1.0,
    use_parallel: bool = True,
    get_history: bool = False
) -> Dict[int, Tuple[chex.Array, chex.Array]]:
    """
    Run bootstrap replications for a joint model training using process-based parallelization.
    """
    control_static = tuple(experiments_control)
    treatment_static = tuple(experiments_treatment)
    seeds = list(range(n_bootstrap))
    if use_parallel:
        with ProcessPoolExecutor() as executor:
            results = {seed: result for seed, result in zip(
                seeds,
                executor.map(
                    _joint_bootstrap_run,
                    seeds,
                    [control_static] * n_bootstrap,
                    [treatment_static] * n_bootstrap,
                    [n_params] * n_bootstrap,
                    [init_theta_sampler] * n_bootstrap,
                    [init_delta_sampler] * n_bootstrap,
                    [agent] * n_bootstrap,
                    [learning_rate] * n_bootstrap,
                    [num_steps] * n_bootstrap,
                    [n_restarts] * n_bootstrap,
                    [min_num_converged] * n_bootstrap,
                    [early_stopping] * n_bootstrap,
                    [delta_penalty_sigma] * n_bootstrap,
                    [get_history] * n_bootstrap,
                )
            )}
        return results
    else:
        results = {}
        for seed in seeds:
            results[seed] = _joint_bootstrap_run(
                seed, control_static, treatment_static, n_params, init_theta_sampler,
                init_delta_sampler, agent, learning_rate, num_steps, n_restarts,
                min_num_converged, early_stopping, delta_penalty_sigma, get_history
            )
        return results

###############################################################################
# Bootstrapping functions for Hierarchical Models
###############################################################################

def bootstrap_train_hierarchical(
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    n_bootstrap: int,
    n_params: int,
    init_theta_pop_sampler: Callable[[], chex.Array],
    make_sample_init_theta_subjects: Callable[[int], Callable[[], chex.Array]],
    agent: Callable,
    sigma_prior: float = 1.0,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
    use_parallel: bool = True,
    get_history: bool = False
) -> Dict[int, Tuple[chex.Array, chex.Array]]:
    """
    Run bootstrap replications for a hierarchical model training using process-based parallelization.
    """
    subjects_static = tuple(tuple(subj) for subj in experiments_by_subject)
    seeds = list(range(n_bootstrap))
    if use_parallel:
        with ProcessPoolExecutor() as executor:
            results = {seed: result for seed, result in zip(
                seeds,
                executor.map(
                    _hierarchical_bootstrap_run,
                    seeds,
                    [subjects_static] * n_bootstrap,
                    [n_params] * n_bootstrap,
                    [init_theta_pop_sampler] * n_bootstrap,
                    [make_sample_init_theta_subjects] * n_bootstrap,
                    [agent] * n_bootstrap,
                    [sigma_prior] * n_bootstrap,
                    [learning_rate] * n_bootstrap,
                    [num_steps] * n_bootstrap,
                    [n_restarts] * n_bootstrap,
                    [min_num_converged] * n_bootstrap,
                    [early_stopping] * n_bootstrap,
                    [get_history] * n_bootstrap,
                )
            )}
        return results
    else:
        results = {}
        for seed in seeds:
            results[seed] = _hierarchical_bootstrap_run(
                seed, subjects_static, n_params, init_theta_pop_sampler,
                make_sample_init_theta_subjects, agent, sigma_prior, learning_rate,
                num_steps, n_restarts, min_num_converged, early_stopping, get_history
            )
        return results
