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
from concurrent.futures import ProcessPoolExecutor


def k_fold_split_experiments(
    experiments: List[Tuple[chex.Array, chex.Array]], k: int
) -> List[Tuple[List[int], List[int]]]:
    """
    Split experiments into K folds.

    :param experiments: List of experiments (each a tuple of choices, rewards).
    :type experiments: List[Tuple[chex.Array, chex.Array]]
    :param k: Number of folds.
    :type k: int
    :returns: List of tuples (train_indices, test_indices) for each fold.
    :rtype: List[Tuple[List[int], List[int]]]
    :note: The experiments are randomly shuffled before splitting.
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
    Split subject experiment lists into K folds.

    :param subject_experiments: List where each element is the list of experiments for one subject.
    :type subject_experiments: List[List[Tuple[chex.Array, chex.Array]]]
    :param k: Number of folds.
    :type k: int
    :returns: List of tuples (train_subject_indices, test_subject_indices).
    :rtype: List[Tuple[List[int], List[int]]]
    :note: Subjects are randomly shuffled before splitting.
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
    get_history: bool = False,
) -> Tuple[float, Dict[int, float], Dict[int, chex.Array]]:
    """
    Perform k-fold cross-validation on the base model.

    For each fold:
      - Split the experiments into training and test sets.
      - Train the model (using multi-start training) on the training experiments.
      - Evaluate predictive performance (negative log-likelihood) on the test experiments.
    
    :param experiments: List of experiments for training.
    :param k: Number of folds.
    :param init_param_sampler: Function returning a new initial parameter vector.
    :param agent: The model function.
    :param learning_rate: Training learning rate.
    :param num_steps: Maximum steps per training run.
    :param n_restarts: Number of random training restarts.
    :param min_num_converged: Minimum number of runs that must converge for early stopping.
    :param early_stopping: Dictionary with individual stopping criteria.
    :param get_history: If True, also return training history.
    :returns: Tuple containing total predictive log-likelihood, per-experiment log-likelihood mapping,
              and fold best parameters mapping.
    :rtype: Tuple[float, Dict[int, float], Dict[int, chex.Array]]
    :note: Predictive log-likelihood is computed as the negative of the negative log-likelihood.
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
        if get_history:
            best_params, _, history = multi_start_train(
                n_restarts=n_restarts,
                init_param_sampler=init_param_sampler,
                agent=agent,
                training_experiments=train_exps,
                learning_rate=learning_rate,
                num_steps=num_steps,
                min_num_converged=min_num_converged,
                early_stopping=early_stopping,
                progress_bar=True,
                verbose=True,
                get_history=True,
            )
        else:
            best_params, _ = multi_start_train(
                n_restarts=n_restarts,
                init_param_sampler=init_param_sampler,
                agent=agent,
                training_experiments=train_exps,
                learning_rate=learning_rate,
                num_steps=num_steps,
                min_num_converged=min_num_converged,
                early_stopping=early_stopping,
                progress_bar=True,
                verbose=True,
                get_history=False,
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

    if get_history:
        return total_pred_ll, per_experiment_ll, fold_params, history
    else:
        return total_pred_ll, per_experiment_ll, fold_params


def run_cv_fold(fold_data):
    """
    Run one cross-validation fold.
    
    :param fold_data: Tuple containing fold index, training experiments, test experiments,
                      initial parameter sampler, agent, learning rate, number of steps,
                      number of restarts, minimum converged count, early stopping and get_history flag.
    :returns: Tuple containing fold index, fold log-likelihood, per-experiment log-likelihood,
              best parameters, and (optionally) history.
    :note: This helper is used for parallel cross-validation.
    """
    (fold_idx, train_exps, test_exps, init_param_sampler, agent,
     learning_rate, num_steps, n_restarts, min_num_converged, early_stopping, get_history) = fold_data

    # Call your multi_start_train function on the training set.
    if get_history:
        best_params, _, history = multi_start_train(
            n_restarts=n_restarts,
            init_param_sampler=init_param_sampler,
            agent=agent,
            training_experiments=train_exps,
            learning_rate=learning_rate,
            num_steps=num_steps,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            progress_bar=False,
            verbose=False,
            get_history=True,
        )
    else:
        best_params, _ = multi_start_train(
            n_restarts=n_restarts,
            init_param_sampler=init_param_sampler,
            agent=agent,
            training_experiments=train_exps,
            learning_rate=learning_rate,
            num_steps=num_steps,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            progress_bar=False,
            verbose=False,
            get_history=False,
        )

    # Evaluate predictive performance on test experiments.
    fold_ll = 0.0
    per_experiment_ll = {}
    for exp_idx, (choices, rewards) in test_exps:
        ll = -float(negative_log_likelihood_experiment(best_params, agent, choices, rewards))
        per_experiment_ll[exp_idx] = ll
        fold_ll += ll
    print(f"Fold {fold_idx} done, fold_ll = {fold_ll:.4f}")

    if get_history:
        return fold_idx, fold_ll, per_experiment_ll, best_params, history
    else:
        return fold_idx, fold_ll, per_experiment_ll, best_params

def parallel_k_fold_cross_validation_train(
    experiments: List[Tuple[chex.Array, chex.Array]],
    k: int,
    init_param_sampler: Callable[[], chex.Array],
    agent: Callable,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
    get_history: bool = False,
) -> Union[Tuple[float, Dict[int, float], Dict[int, chex.Array]],
           Tuple[float, Dict[int, float], Dict[int, chex.Array], Dict[int, chex.Array]]]:
    """
    Perform k-fold cross-validation in parallel for the base model.

    :param experiments: List of experiments.
    :param k: Number of folds.
    :param init_param_sampler: Initial parameter sampler.
    :param agent: Model function.
    :param learning_rate: Learning rate.
    :param num_steps: Number of training steps.
    :param n_restarts: Number of restarts.
    :param min_num_converged: Minimum converged runs.
    :param early_stopping: Early stopping parameters.
    :param get_history: Flag for returning training history.
    :returns: Tuple with total predictive log-likelihood, per-experiment likelihood dictionary,
              and fold parameters dictionary.
    """
    splits = k_fold_split_experiments(experiments, k)
    cv_fold_data = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_exps = [experiments[i] for i in train_idx]
        test_exps = [(i, experiments[i]) for i in test_idx]
        cv_fold_data.append((fold_idx, train_exps, test_exps,
                             init_param_sampler, agent, learning_rate,
                             num_steps, n_restarts, min_num_converged, early_stopping, get_history))
    
    total_pred_ll = 0.0
    per_experiment_ll = {}
    fold_params = {}
    fold_history = {}

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_cv_fold, cv_fold_data))

    if get_history:
        for fold_idx, fold_ll, fold_exp_ll, best_params, history in results:
            total_pred_ll += fold_ll
            per_experiment_ll.update(fold_exp_ll)
            fold_params[fold_idx] = best_params
            fold_history[fold_idx] = history
        print(f"\nTotal predictive log-likelihood (across folds): {total_pred_ll:.4f}")
        return total_pred_ll, per_experiment_ll, fold_params, fold_history
    else:
        for fold_idx, fold_ll, fold_exp_ll, best_params in results:
            total_pred_ll += fold_ll
            per_experiment_ll.update(fold_exp_ll)
            fold_params[fold_idx] = best_params
        print(f"\nTotal predictive log-likelihood (across folds): {total_pred_ll:.4f}")
        return total_pred_ll, per_experiment_ll, fold_params


def k_fold_cross_validation_train_joint(
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_treatment: List[Tuple[chex.Array, chex.Array]],
    k: int,
    n_params: int,
    init_theta_sampler: Callable[[], chex.Array],
    init_delta_sampler: Callable[[], chex.Array],
    agent: Callable,  # e.g., your joint RL model
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
    delta_penalty_sigma: float = 1.0,
    get_history: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Tuple[chex.Array, chex.Array]]]:
    """
    Perform k-fold cross-validation for the joint model.

    For each fold:
      - Independently split the control and experimental experiments.
      - Train the joint model (via multi-start joint training) on the training sets.
      - Evaluate predictive log-likelihood on the held-out test sets.

    :param experiments_control: List of control experiments (each a (choices, rewards) tuple).
    :param experiments_treatment: List of experimental experiments.
    :param k: Number of folds.
    :param n_params: Number of parameters in the model.
    :param init_theta_sampler: Function that returns a new initial theta (for the control group).
    :param init_delta_sampler: Function that returns a new initial delta.
    :param agent: The joint agent model function.
    :param learning_rate: Learning rate (default 5e-2).
    :param num_steps: Maximum training steps (default 10,000).
    :param n_restarts: Number of random restarts for multi-start training.
    :param min_num_converged: Minimum number of runs that must converge to the best loss.
    :param early_stopping: Optional dictionary with early stopping parameters.
    :param delta_penalty_sigma: Penalty for the delta norm (default 1.0).
    :param get_history: Whether to return the training history.
    :returns: Tuple containing total predictive log-likelihood, per-experiment log-likelihood mapping,
              and fold best parameters mapping.
    :rtype: Tuple[float, Dict[str, float], Dict[str, Tuple[chex.Array, chex.Array]]]
    """
    # Create separate folds for control and experimental experiments.
    splits_control = k_fold_split_experiments(experiments_control, k)
    splits_exp = k_fold_split_experiments(experiments_treatment, k)

    total_pred_ll = 0.0
    per_experiment_ll = {}  # keys like "control_3" or "exp_5"
    fold_params = {}  # new: store (best_theta, best_delta) per fold
    fold_history = {}  # new: store training history per fold

    for fold_idx in range(k):
        print(f"\n=== Joint Fold {fold_idx+1}/{k} ===")
        # For control group:
        train_idx_control, test_idx_control = splits_control[fold_idx]
        train_control = [experiments_control[i] for i in train_idx_control]
        test_control = [(i, experiments_control[i]) for i in test_idx_control]
        # For experimental group:
        train_idx_exp, test_idx_exp = splits_exp[fold_idx]
        train_exp = [experiments_treatment[i] for i in train_idx_exp]
        test_exp = [(i, experiments_treatment[i]) for i in test_idx_exp]
        
        if get_history:
            # Train joint model on the union of control and experimental training sets.
            best_theta, best_delta, _, history = multi_start_joint_train(
                init_theta_sampler=init_theta_sampler,
                init_delta_sampler=init_delta_sampler,
                agent=agent,
                n_params=n_params,
                experiments_control=train_control,
                experiments_treatment=train_exp,
                learning_rate=learning_rate,
                num_steps=num_steps,
                n_restarts=n_restarts,
                min_num_converged=min_num_converged,
                early_stopping=early_stopping,
                delta_penalty_sigma=delta_penalty_sigma,
                verbose=True,
                progress_bar=True,
                get_history=True,
            )
            fold_history[fold_idx] = history
        else:
            best_theta, best_delta, _ = multi_start_joint_train(
                init_theta_sampler=init_theta_sampler,
                init_delta_sampler=init_delta_sampler,
                agent=agent,
                n_params=n_params,
                experiments_control=train_control,
                experiments_treatment=train_exp,
                learning_rate=learning_rate,
                num_steps=num_steps,
                n_restarts=n_restarts,
                min_num_converged=min_num_converged,
                early_stopping=early_stopping,
                delta_penalty_sigma=delta_penalty_sigma,
                verbose=True,
                progress_bar=True,
                get_history=False,
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

    if get_history:
        return total_pred_ll, per_experiment_ll, fold_params, fold_history
    else:
        return total_pred_ll, per_experiment_ll, fold_params

def run_cv_fold_joint(fold_data):
    """
    Run one joint cross-validation fold.
    
    :param fold_data: Tuple containing fold index, training control experiments, test control experiments,
                      training experimental experiments, test experimental experiments, initial theta sampler,
                      initial delta sampler, agent, learning rate, number of steps, number of restarts,
                      minimum converged count, early stopping, number of parameters, delta penalty sigma,
                      and get_history flag.
    :returns: Tuple containing fold index, fold log-likelihood, per-experiment log-likelihood,
              best theta, best delta, and (optionally) history.
    :note: This helper is used for parallel joint cross-validation.
    """
    (fold_idx, train_control, test_control, train_exp, test_exp,
     init_theta_sampler, init_delta_sampler, agent, learning_rate,
     num_steps, n_restarts, min_num_converged, early_stopping, 
     n_params, delta_penalty_sigma, get_history) = fold_data

    if get_history:
        best_theta, best_delta, _, history = multi_start_joint_train(
            n_restarts=n_restarts,
            init_theta_sampler=init_theta_sampler,
            init_delta_sampler=init_delta_sampler,
            agent=agent,
            n_params=n_params,
            experiments_control=train_control,
            experiments_treatment=train_exp,
            learning_rate=learning_rate,
            num_steps=num_steps,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            verbose=False,
            progress_bar=False,
            get_history=True,
            delta_penalty_sigma=delta_penalty_sigma,
        )
    else:
        best_theta, best_delta, _ = multi_start_joint_train(
            n_restarts=n_restarts,
            init_theta_sampler=init_theta_sampler,
            init_delta_sampler=init_delta_sampler,
            agent=agent,
            n_params=n_params,
            experiments_control=train_control,
            experiments_treatment=train_exp,
            learning_rate=learning_rate,
            num_steps=num_steps,
            min_num_converged=min_num_converged,
            early_stopping=early_stopping,
            verbose=False,
            progress_bar=False,
            delta_penalty_sigma=delta_penalty_sigma,
        )

    fold_ll = 0.0
    per_experiment_ll = {}
    # Evaluate control test set.
    for exp_idx, (choices, rewards) in test_control:
        ll = -float(negative_log_likelihood_experiment(best_theta, agent, choices, rewards))
        per_experiment_ll[f"control_{exp_idx}"] = ll
        fold_ll += ll
    # Evaluate experimental test set.
    predicted_params_exp = best_theta + best_delta
    for exp_idx, (choices, rewards) in test_exp:
        ll = -float(negative_log_likelihood_experiment(predicted_params_exp, agent, choices, rewards))
        per_experiment_ll[f"exp_{exp_idx}"] = ll
        fold_ll += ll
    print(f"Joint fold {fold_idx} done, fold_ll = {fold_ll:.4f}")

    if get_history:
        return fold_idx, fold_ll, per_experiment_ll, best_theta, best_delta, history
    else:
        return fold_idx, fold_ll, per_experiment_ll, best_theta, best_delta

def parallel_k_fold_cross_validation_train_joint(
    experiments_control: List[Tuple[chex.Array, chex.Array]],
    experiments_treatment: List[Tuple[chex.Array, chex.Array]],
    k: int,
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
    get_history: bool = False,
) -> Tuple[float, Dict[str, float], Dict[int, Tuple[chex.Array, chex.Array]]]:
    """
    Perform k-fold cross-validation in parallel for the joint model.

    :param experiments_control: List of control experiments.
    :param experiments_treatment: List of experimental experiments.
    :param k: Number of folds.
    :param n_params: Number of parameters in the model.
    :param init_theta_sampler: Initial theta sampler.
    :param init_delta_sampler: Initial delta sampler.
    :param agent: Model function.
    :param learning_rate: Learning rate.
    :param num_steps: Number of training steps.
    :param n_restarts: Number of restarts.
    :param min_num_converged: Minimum converged runs.
    :param early_stopping: Early stopping parameters.
    :param delta_penalty_sigma: Penalty for delta norm.
    :param get_history: Flag for returning training history.
    :returns: Tuple with total predictive log-likelihood, per-experiment likelihood dictionary,
              and fold parameters dictionary.
    """
    splits_control = k_fold_split_experiments(experiments_control, k)
    splits_exp = k_fold_split_experiments(experiments_treatment, k)
    cv_fold_data = []

    for fold_idx in range(k):
        train_idx_control, test_idx_control = splits_control[fold_idx]
        train_control = [experiments_control[i] for i in train_idx_control]
        test_control = [(i, experiments_control[i]) for i in test_idx_control]
        train_idx_exp, test_idx_exp = splits_exp[fold_idx]
        train_exp = [experiments_treatment[i] for i in train_idx_exp]
        test_exp = [(i, experiments_treatment[i]) for i in test_idx_exp]
        cv_fold_data.append((fold_idx, train_control, test_control, train_exp, test_exp,
                             init_theta_sampler, init_delta_sampler, agent, learning_rate,
                             num_steps, n_restarts, min_num_converged, early_stopping, 
                             n_params, delta_penalty_sigma, get_history))
    
    total_pred_ll = 0.0
    per_experiment_ll = {}
    fold_params = {}
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_cv_fold_joint, cv_fold_data))
    
    if get_history:
        fold_history = {}
        for fold_idx, fold_ll, fold_exp_ll, best_theta, best_delta, history in results:
            total_pred_ll += fold_ll
            per_experiment_ll.update(fold_exp_ll)
            fold_params[fold_idx] = (best_theta, best_delta)
            fold_history[fold_idx] = history
        print(f"\nTotal joint predictive log-likelihood: {total_pred_ll:.4f}")
        return total_pred_ll, per_experiment_ll, fold_params, fold_history
    else:
        for fold_idx, fold_ll, fold_exp_ll, best_theta, best_delta in results:
            total_pred_ll += fold_ll
            per_experiment_ll.update(fold_exp_ll)
            fold_params[fold_idx] = (best_theta, best_delta)
        print(f"\nTotal joint predictive log-likelihood: {total_pred_ll:.4f}")
        return total_pred_ll, per_experiment_ll, fold_params


def k_fold_cross_validation_train_hierarchical(
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    k: int,
    n_params: int,
    init_theta_pop_sampler: Callable[[], chex.Array],
    make_sample_init_theta_subjects: Callable[[], chex.Array],
    agent: Callable,  # hierarchical agent model (signature: (params, agent_state, choice, reward))
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
    sigma_prior: float = 1.0,
    get_history: bool = False,
) -> Union[Tuple[float, Dict[int, float], Dict[int, Tuple[chex.Array, chex.Array]],
        Tuple[float, Dict[int, float], Dict[int, Tuple[chex.Array, chex.Array]], Dict[int, chex.Array]]]]:
    """
    Perform k-fold cross-validation for hierarchical model training.
    
    For each fold:
      - Split subjects (i.e. the list of subject experiment lists) into training and test groups.
      - Train the hierarchical model (using multi-start hierarchical training) on the training subjects.
      - For each test subject, set the predicted subject parameter equal to the population parameter
        (i.e. using the population parameter as the prediction) and compute its predictive log‑likelihood.
    
    :param experiments_by_subject: List of subject experiment lists.
    :param k: Number of folds.
    :param n_params: Number of parameters in the model.
    :param init_theta_pop_sampler: Initial population parameter sampler.
    :param make_sample_init_theta_subjects: Function to create a local sampler for subject-specific parameters.
    :param agent: Hierarchical agent model function.
    :param learning_rate: Learning rate.
    :param num_steps: Number of training steps.
    :param n_restarts: Number of restarts.
    :param min_num_converged: Minimum converged runs.
    :param early_stopping: Early stopping parameters.
    :param sigma_prior: Prior standard deviation for the hierarchical model.
    :param get_history: Flag for returning training history.
    :returns: Tuple with total predictive log-likelihood, per-subject likelihood dictionary,
              and fold parameters dictionary.
    """
    splits = k_fold_split_subjects(experiments_by_subject, k)
    total_pred_ll = 0.0
    per_subject_ll = {}  # keys will be subject indices
    fold_params = {}  # new: store fitted hierarchical parameters per fold
    fold_history = {}  # new: store training history per fold

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Hierarchical Fold {fold_idx+1}/{k} ===")
        # Build training and test sets (each element is the list of experiments for that subject)
        train_exps = [experiments_by_subject[i] for i in train_idx]
        test_exps = [(i, experiments_by_subject[i]) for i in test_idx]

        # Get the number of training subjects for this fold:
        n_train_subjects = len(train_exps)

        # Create a local sampler for subject-specific parameters:
        local_sample_init_theta_subjects = make_sample_init_theta_subjects(n_train_subjects)

        # Now call multi_start_hierarchical_train with the local sampler:
        if get_history:
            best_theta_pop, best_theta_subjects, _, history = multi_start_hierarchical_train(
                n_params=n_params,
                n_restarts=n_restarts,
                init_theta_pop_sampler=init_theta_pop_sampler,
                init_theta_subjects_sampler=local_sample_init_theta_subjects,
                agent=agent,
                experiments_by_subject=train_exps,
                learning_rate=learning_rate,
                num_steps=num_steps,
                sigma_prior=sigma_prior,
                verbose=True,
                early_stopping=early_stopping,
                min_num_converged=min_num_converged,
                progress_bar=False,
                get_history=True,
            )
            fold_history[fold_idx] = history
        else:
            best_theta_pop, best_theta_subjects, _ = multi_start_hierarchical_train(
                n_params=n_params,
                n_restarts=n_restarts,
                init_theta_pop_sampler=init_theta_pop_sampler,
                init_theta_subjects_sampler=local_sample_init_theta_subjects,
                agent=agent,
                experiments_by_subject=train_exps,
                learning_rate=learning_rate,
                num_steps=num_steps,
                sigma_prior=sigma_prior,
                verbose=True,
                early_stopping=early_stopping,
                min_num_converged=min_num_converged,
                progress_bar=False,
                get_history=False,
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

    if get_history:
        return total_pred_ll, per_subject_ll, fold_params, fold_history
    else:
        return total_pred_ll, per_subject_ll, fold_params

def run_cv_fold_hierarchical(fold_data):
    """
    Helper function to run one hierarchical CV fold.
    
    :param fold_data: Tuple containing fold index, training experiments, test experiments,
                      initial population parameter sampler, initial subject parameter sampler,
                      agent, learning rate, number of steps, number of restarts, minimum converged count,
                      early stopping parameters, number of parameters, prior standard deviation, and get_history flag.
    :returns: Tuple containing fold index, fold log-likelihood, per-subject log-likelihood,
              and best parameters (population and subject).
    """
    (fold_idx, train_exps, test_exps, init_theta_pop_sampler, init_theta_subjects_sampler,
     agent, learning_rate, num_steps, n_restarts, min_num_converged, early_stopping,
     n_params, sigma_prior, get_history) = fold_data

    # Train the hierarchical model on the training subjects.
    if get_history:
        best_theta_pop, best_theta_subjects, _, history = multi_start_hierarchical_train(
            n_params=n_params,
            n_restarts=n_restarts,
            init_theta_pop_sampler=init_theta_pop_sampler,
            init_theta_subjects_sampler=init_theta_subjects_sampler,
            agent=agent,
            experiments_by_subject=train_exps,
            learning_rate=learning_rate,
            num_steps=num_steps,
            sigma_prior=sigma_prior,
            verbose=False,
            early_stopping=early_stopping,
            min_num_converged=min_num_converged,
            progress_bar=False,
            get_history=True,
        )
    else:
        best_theta_pop, best_theta_subjects, _ = multi_start_hierarchical_train(
            n_params=n_params,
            n_restarts=n_restarts,
            init_theta_pop_sampler=init_theta_pop_sampler,
            init_theta_subjects_sampler=init_theta_subjects_sampler,
            agent=agent,
            experiments_by_subject=train_exps,
            learning_rate=learning_rate,
            num_steps=num_steps,
            sigma_prior=sigma_prior,
            verbose=False,
            early_stopping=early_stopping,
            min_num_converged=min_num_converged,
            progress_bar=False,
            get_history=False,
        )

    # For each test subject, use the population parameter as the predicted subject parameter.
    fold_ll = 0.0
    per_subject_ll = {}
    for subj_idx, exps in test_exps:
        # Prediction: subject parameter = best_theta_pop
        ll = -float(total_negative_log_likelihood(best_theta_pop, agent, exps))
        per_subject_ll[subj_idx] = ll
        fold_ll += ll

    print(f"Hierarchical fold {fold_idx} done, fold_ll = {fold_ll:.4f}")
    if get_history:
        return fold_idx, fold_ll, per_subject_ll, best_theta_pop, best_theta_subjects, history
    else:
        return fold_idx, fold_ll, per_subject_ll, best_theta_pop, best_theta_subjects


def parallel_k_fold_cross_validation_train_hierarchical(
    experiments_by_subject: List[List[Tuple[chex.Array, chex.Array]]],
    k: int,
    n_params: int,
    init_theta_pop_sampler: Callable[[], chex.Array],
    make_sample_init_theta_subjects: Callable[[], chex.Array],
    agent: Callable,  # hierarchical agent model (signature: (params, agent_state, choice, reward))
    sigma_prior: float = 1.0,
    learning_rate: float = 5e-2,
    num_steps: int = 10000,
    n_restarts: int = 10,
    min_num_converged: int = 3,
    early_stopping: Optional[Dict[str, float]] = None,
    get_history: bool = False,
) -> Union[Tuple[float, Dict[int, float], Dict[int, Tuple[chex.Array, chex.Array]],
        Tuple[float, Dict[int, float], Dict[int, Tuple[chex.Array, chex.Array]], Dict[int, chex.Array]]]]:
    """
    Perform k-fold cross-validation for hierarchical model training in parallel.

    For each fold:
      - Split subjects into training and test groups.
      - Train the hierarchical model (via multi_start_hierarchical_train) on the training subjects.
      - For each test subject, predict using the population parameter and compute its predictive log‑likelihood.

    :param experiments_by_subject: List of subject experiment lists.
    :param k: Number of folds.
    :param n_params: Number of parameters in the model.
    :param init_theta_pop_sampler: Initial population parameter sampler.
    :param make_sample_init_theta_subjects: Function to create a local sampler for subject-specific parameters.
    :param agent: Hierarchical agent model function.
    :param sigma_prior: Prior standard deviation for the hierarchical model.
    :param learning_rate: Learning rate.
    :param num_steps: Number of training steps.
    :param n_restarts: Number of restarts.
    :param min_num_converged: Minimum converged runs.
    :param early_stopping: Early stopping parameters.
    :returns: Tuple with total predictive log-likelihood, per-subject likelihood dictionary,
              and fold parameters dictionary.
    """
    splits = k_fold_split_subjects(experiments_by_subject, k)
    cv_fold_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_exps = [experiments_by_subject[i] for i in train_idx]
        test_exps = [(i, experiments_by_subject[i]) for i in test_idx]
        
        # Create a local sampler for subject-specific parameters:
        local_sample_init_theta_subjects = make_sample_init_theta_subjects(len(train_exps))

        cv_fold_data.append((fold_idx, train_exps, test_exps,
                             init_theta_pop_sampler, local_sample_init_theta_subjects,
                             agent, learning_rate, num_steps, n_restarts,
                             min_num_converged, early_stopping,
                             n_params, sigma_prior, get_history))

    total_pred_ll = 0.0
    per_subject_ll = {}
    fold_params = {}
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_cv_fold_hierarchical, cv_fold_data))

    if get_history:
        fold_history = {}
        for fold_idx, fold_ll, fold_subj_ll, best_theta_pop, best_theta_subjects, history in results:
            total_pred_ll += fold_ll
            per_subject_ll.update(fold_subj_ll)
            fold_params[fold_idx] = (best_theta_pop, best_theta_subjects)
            fold_history[fold_idx] = history
        print(f"\nTotal hierarchical predictive log-likelihood: {total_pred_ll:.4f}")
        return total_pred_ll, per_subject_ll, fold_params, fold_history
    else:
        for fold_idx, fold_ll, fold_subj_ll, params in results:
            total_pred_ll += fold_ll
            per_subject_ll.update(fold_subj_ll)
            fold_params[fold_idx] = params
        print(f"\nTotal hierarchical predictive log-likelihood: {total_pred_ll:.4f}")
        return total_pred_ll, per_subject_ll, fold_params
