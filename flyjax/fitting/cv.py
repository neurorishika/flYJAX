# flyjax/fitting/cv.py
import numpy as np
import jax
import jax.numpy as jnp
import chex
from typing import List, Tuple, Callable, Dict, Optional
from tqdm.auto import trange
from flyjax.fitting.train import multi_start_train
from flyjax.fitting.joint import multi_start_joint_train
from flyjax.fitting.hierarchical import multi_start_hierarchical_train
from flyjax.fitting.evaluation import total_negative_log_likelihood, negative_log_likelihood_experiment


def k_fold_split_experiments(
    experiments: List[Tuple[chex.Array, chex.Array]], k: int
) -> List[Tuple[List[int], List[int]]]:
    """
    Split experiments into K folds and return a list of (train_indices, test_indices).
    """
    n = len(experiments)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        test_idx = folds[i].tolist()
        train_idx = np.hstack([folds[j] for j in range(k) if j != i]).tolist()
        splits.append((train_idx, test_idx))
    return splits

def k_fold_split_subjects(
    subject_experiments: List[List[Tuple[chex.Array, chex.Array]]],
    k: int
) -> List[Tuple[List[int], List[int]]]:
    """
    Split subjects (i.e. the list of experiment lists) into k folds.
    Returns a list of (train_subject_indices, test_subject_indices).
    """
    n = len(subject_experiments)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        test_idx = folds[i].tolist()
        train_idx = np.hstack([folds[j] for j in range(k) if j != i]).tolist()
        splits.append((train_idx, test_idx))
    return splits


def k_fold_cross_validation_train(
    experiments: List[Tuple[chex.Array, chex.Array]],
    k: int,
    init_param_sampler: Callable[[], chex.Array],
    agent: Callable,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[int, float], Dict[int, chex.Array]]:
    """
    Perform k-fold cross-validation for model training.

    For each fold:
      - Train the model (using multi-start training) on the training set.
      - Evaluate predictive log-likelihood on the held-out test experiments.

    Parameters:
      experiments: List of experiments. Each experiment is a tuple (choices, rewards)
                   corresponding to one fly.
      k: Number of folds.
      init_param_sampler: Function to sample initial parameters.
      agent: The model function (signature: (params, agent_state, choice, reward)).
      learning_rate: Learning rate for training.
      num_steps: Number of training steps.
      n_restarts: Number of random restarts for multi-start training.
      min_num_converged: Minimum number of runs that must converge to the best loss.
      early_stopping: Optional early stopping parameters.

    Returns:
      total_pred_ll: Total predictive log-likelihood summed over all test experiments.
      per_experiment_ll: Dictionary mapping experiment (fly) index to its predictive log-likelihood.
      fold_params: Dictionary mapping fold index to the best fitted parameters from that fold.
    """
    splits = k_fold_split_experiments(experiments, k)
    total_pred_ll = 0.0
    per_experiment_ll = {}  # keys are experiment indices
    fold_params = {}  # new: store best_params for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Fold {fold_idx+1}/{k} ===")
        # Build training and test sets
        train_exps = [experiments[i] for i in train_idx]
        test_exps = [(i, experiments[i]) for i in test_idx]  # store original index with experiment
        
        # Train the model on the training set using multi-start training
        best_params, _ = multi_start_train(
            n_restarts=n_restarts,
            init_param_sampler=init_param_sampler,
            agent=agent,
            training_experiments=train_exps,
            learning_rate=learning_rate,
            num_steps=num_steps,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
        )
        fold_params[fold_idx] = best_params  # save the fitted parameters for this fold

        # Evaluate predictive performance on each test experiment.
        fold_ll = 0.0
        for exp_idx, (choices, rewards) in test_exps:
            # Compute predicted log likelihood: negative NLL.
            ll = -float(
                negative_log_likelihood_experiment(best_params, agent, choices, rewards)
            )
            per_experiment_ll[exp_idx] = ll
            fold_ll += ll
        print(f"Fold {fold_idx+1} predictive log-likelihood: {fold_ll:.4f}")
        total_pred_ll += fold_ll
    print(f"\nTotal predictive log-likelihood (across folds): {total_pred_ll:.4f}")
    return total_pred_ll, per_experiment_ll, fold_params


def k_fold_cross_validation_train_joint(
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_exp: List[Tuple[chex.Array, chex.Array]],
    k: int,
    init_theta_sampler: Callable[[], chex.Array],
    init_delta_sampler: Callable[[], chex.Array],
    agent: Callable,  # e.g., your joint RL model
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, Tuple[chex.Array, chex.Array]]]:
    """
    Perform k-fold cross-validation for the joint model.

    For each fold:
      - Independently split the control and experimental experiments.
      - Train the joint model (via multi-start joint training) on the training sets.
      - Evaluate predictive log-likelihood on the held-out test sets.

    Parameters:
      experiments_control: List of control experiments (each a (choices, rewards) tuple).
      experiments_exp: List of experimental experiments.
      k: Number of folds.
      init_theta_sampler: Function that returns a new initial theta (for the control group).
      init_delta_sampler: Function that returns a new initial delta.
      agent: The joint agent model function.
      learning_rate: Learning rate (default 5e-2).
      num_steps: Maximum training steps (default 10,000).
      n_restarts: Number of random restarts for multi-start training.
      min_num_converged: Minimum number of runs that must converge to the best loss.
      early_stopping: Optional dictionary with early stopping parameters.

    Returns:
      total_pred_ll: Total predictive log-likelihood summed over all test experiments.
      per_experiment_ll: Dictionary mapping experiment indices (prefixed by group) to their predictive log-likelihood.
      fold_params: Dictionary mapping fold index to a tuple (best_theta, best_delta) from that fold.
    """
    # Create separate folds for control and experimental experiments.
    splits_control = k_fold_split_experiments(experiments_control, k)
    splits_exp = k_fold_split_experiments(experiments_exp, k)

    total_pred_ll = 0.0
    per_experiment_ll = {}  # keys like "control_3" or "exp_5"
    fold_params = {}  # new: store (best_theta, best_delta) per fold

    for fold_idx in range(k):
        print(f"\n=== Joint Fold {fold_idx+1}/{k} ===")
        # For control group:
        train_idx_control, test_idx_control = splits_control[fold_idx]
        train_control = [experiments_control[i] for i in train_idx_control]
        test_control = [(i, experiments_control[i]) for i in test_idx_control]
        # For experimental group:
        train_idx_exp, test_idx_exp = splits_exp[fold_idx]
        train_exp = [experiments_exp[i] for i in train_idx_exp]
        test_exp = [(i, experiments_exp[i]) for i in test_idx_exp]

        # Train joint model on the union of control and experimental training sets.
        best_theta, best_delta, _ = multi_start_joint_train(
            init_theta_sampler=init_theta_sampler,
            init_delta_sampler=init_delta_sampler,
            agent=agent,
            experiments_control=train_control,
            experiments_exp=train_exp,
            learning_rate=learning_rate,
            num_steps=num_steps,
            n_restarts=n_restarts,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            verbose=True,
        )
        fold_params[fold_idx] = (best_theta, best_delta)  # store fitted parameters

        fold_ll = 0.0
        # Evaluate on control test experiments (using best_theta for control).
        for exp_idx, (choices, rewards) in test_control:
            ll = -float(
                negative_log_likelihood_experiment(best_theta, agent, choices, rewards)
            )
            per_experiment_ll[f"control_{exp_idx}"] = ll
            fold_ll += ll
        # Evaluate on experimental test experiments (predicted parameters = best_theta + best_delta).
        predicted_params_exp = best_theta + best_delta
        for exp_idx, (choices, rewards) in test_exp:
            ll = -float(
                negative_log_likelihood_experiment(
                    predicted_params_exp, agent, choices, rewards
                )
            )
            per_experiment_ll[f"exp_{exp_idx}"] = ll
            fold_ll += ll

        print(f"Fold {fold_idx+1} predictive log-likelihood: {fold_ll:.4f}")
        total_pred_ll += fold_ll

    print(f"\nTotal joint predictive log-likelihood (across folds): {total_pred_ll:.4f}")
    return total_pred_ll, per_experiment_ll, fold_params


def k_fold_cross_validation_train_hierarchical(
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    k: int,
    init_theta_pop_sampler: Callable[[], chex.Array],
    init_theta_subjects_sampler: Callable[[], chex.Array],
    agent: Callable,  # hierarchical agent model (signature: (params, agent_state, choice, reward))
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[int, float], Dict[int, Tuple[chex.Array, chex.Array]]]:
    """
    Perform k-fold cross-validation for hierarchical model training.
    
    For each fold:
      - Split subjects (i.e. the list of subject experiment lists) into training and test groups.
      - Train the hierarchical model (using multi-start hierarchical training) on the training subjects.
      - For each test subject, set the predicted subject parameter equal to the population parameter
        (i.e. using the population parameter as the prediction) and compute its predictive log‑likelihood.
    
    Returns:
      total_pred_ll: Total predictive log‑likelihood summed over all test subjects.
      per_subject_ll: Dictionary mapping subject (fly) index to its predictive log‑likelihood.
      fold_params: Dictionary mapping fold index to a tuple (best_theta_pop, best_theta_subjects)
                   from that fold.
    """
    splits = k_fold_split_subjects(experiments_by_subject, k)
    total_pred_ll = 0.0
    per_subject_ll = {}  # keys will be subject indices
    fold_params = {}  # new: store fitted hierarchical parameters per fold

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Hierarchical Fold {fold_idx+1}/{k} ===")
        # Build training and test sets (each element is the list of experiments for that subject)
        train_exps = [experiments_by_subject[i] for i in train_idx]
        test_exps = [(i, experiments_by_subject[i]) for i in test_idx]
        
        # Train the hierarchical model on the training subjects.
        best_theta_pop, best_theta_subjects, _ = multi_start_hierarchical_train(
            n_restarts=n_restarts,
            init_theta_pop_sampler=init_theta_pop_sampler,
            init_theta_subjects_sampler=init_theta_subjects_sampler,
            agent=agent,
            experiments_by_subject=train_exps,
            learning_rate=learning_rate,
            num_steps=num_steps,
            sigma_prior=1.0,
            verbose=True,
            early_stopping=early_stopping,
            min_num_converged=min_num_converged,
        )
        fold_params[fold_idx] = (best_theta_pop, best_theta_subjects)  # store parameters

        # For each held-out subject, use best_theta_pop as prediction.
        fold_ll = 0.0
        for subj_idx, exps in test_exps:
            ll = -float(total_negative_log_likelihood(best_theta_pop, agent, exps))
            per_subject_ll[subj_idx] = ll
            fold_ll += ll
        print(f"Fold {fold_idx+1} predictive log-likelihood: {fold_ll:.4f}")
        total_pred_ll += fold_ll

    print(f"\nTotal hierarchical predictive log-likelihood (across folds): {total_pred_ll:.4f}")
    return total_pred_ll, per_subject_ll, fold_params
