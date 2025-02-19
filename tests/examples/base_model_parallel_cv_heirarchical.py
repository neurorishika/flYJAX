import jax
import jax.numpy as jnp
import numpy as np
import chex
from flyjax.agent.model import test_agent
from flyjax.simulation.simulate import simulate_experiment_jit, simulate_dataset_jit, simulate_dataset_jit_different_params
from flyjax.simulation.parse import parse_reward_matrix
from flyjax.fitting.cv import k_fold_cross_validation_train_hierarchical, \
    parallel_k_fold_cross_validation_train_hierarchical
from flyjax.fitting.hierarchical import evaluate_hierarchical_model
from flyjax.utils.plotting import plot_single_experiment_data
from typing import Tuple, Dict, Any, List, Callable, Optional
from flyjax.fitting.samplers import base_randn_sampler, make_base_randn_sampler
from functools import partial

if __name__ == "__main__":
    # 0. Set up random number generator
    rng_key = jax.random.PRNGKey(0)
        
    # 1. Define "true" parameters
    true_params = jnp.array([2.0, -1.0, 1.0, 0.1])
    print("True params:", true_params)

    # --- Set true parameters for simulation ---
    # params = [alpha_learn_logit, alpha_forget_logit, kappa_reward, kappa_omission]
    true_params = jnp.array([2.0, -1.0, 1.0, 0.1])
    print("True parameters:", true_params)

    # --- Simulate a dataset ---
    # Define the reward probabilities for each experiment.
    n_replicates = 8
    reward_matrices = [
        parse_reward_matrix("[0.0,0.0]x30;[0.33,0.17]x40;[0.33,0.67]x40;[0.17,0.33]x40;[0.89,0.11]x40;[0.33,0.17]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.33,0.17]x40;[0.33,0.67]x40;[0.89,0.11]x40;[0.67,0.33]x40;[0.17,0.33]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.33,0.17]x40;[0.11,0.89]x40;[0.17,0.33]x40;[0.67,0.33]x40;[0.33,0.17]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.33,0.17]x40;[0.11,0.89]x40;[0.67,0.33]x40;[0.89,0.11]x40;[0.17,0.33]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.67,0.33]x40;[0.17,0.33]x40;[0.33,0.67]x40;[0.89,0.11]x40;[0.67,0.33]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.67,0.33]x40;[0.17,0.33]x40;[0.89,0.11]x40;[0.33,0.17]x40;[0.33,0.67]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.67,0.33]x40;[0.11,0.89]x40;[0.33,0.17]x40;[0.89,0.11]x40;[0.33,0.67]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.67,0.33]x40;[0.11,0.89]x40;[0.33,0.67]x40;[0.89,0.11]x40;[0.67,0.33]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.89,0.11]x40;[0.17,0.33]x40;[0.11,0.89]x40;[0.67,0.33]x40;[0.89,0.11]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.89,0.11]x40;[0.17,0.33]x40;[0.67,0.33]x40;[0.33,0.17]x40;[0.11,0.89]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.89,0.11]x40;[0.33,0.67]x40;[0.33,0.17]x40;[0.67,0.33]x40;[0.11,0.89]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.89,0.11]x40;[0.33,0.67]x40;[0.11,0.89]x40;[0.33,0.17]x40;[0.89,0.11]x40")
    ]*n_replicates

    rng_key = jax.random.PRNGKey(0)
    # --- Multi subject training ---

    # build a dataset with multiple subjects assuming each subject has the different parameters sampled from a normal distribution
    n_subjects = len(reward_matrices)
    subject_params = []
    for _ in range(n_subjects):
        rng_key, subkey = jax.random.split(rng_key)
        subject_params.append(true_params + 0.2*jax.random.normal(rng_key, shape=true_params.shape))

    # Simulate the dataset for each subject.
    rng_key, subkey = jax.random.split(rng_key)
    choices, rewards = simulate_dataset_jit_different_params(jnp.stack(subject_params), jnp.stack(reward_matrices), test_agent, rng_key, baiting=True)
    # Assuming 'choices' and 'rewards' are produced by simulate_dataset_jit_different_params
    n_experiments = choices.shape[0]
    subject_experiments = [[(np.array(choices[i]), np.array(rewards[i]))] for i in range(n_experiments)] # this is done because we assume each subject has only one experiment

    print("Number of subjects:", n_subjects)
    print("True parameters for the subjects:", subject_params)

    # --- Hierarchical model fitting ---
    init_theta_pop_sampler = partial(base_randn_sampler, mean=0.0, std=1.0, n_params=4)
    make_sample_init_theta_subjects = partial(make_base_randn_sampler, mean=0.0, std=1.0, n_params=4)

    total_pred_ll, per_experiment_ll, params = k_fold_cross_validation_train_hierarchical(
        experiments_by_subject=subject_experiments,
        init_theta_pop_sampler=init_theta_pop_sampler,
        make_sample_init_theta_subjects=make_sample_init_theta_subjects,
        k=5,
        agent=test_agent,
        learning_rate=5e-2,
        num_steps=10000,
        n_restarts=10,
        min_num_converged=3,
        early_stopping={
            "min_delta": 1e-4,
        }
    )

    print("Estimated parameters:", params)
    print("Total predicted log likelihood:", total_pred_ll)
    print("Per experiment log likelihood:", per_experiment_ll)



