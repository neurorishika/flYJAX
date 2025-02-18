import jax
import jax.numpy as jnp
import numpy as np
import chex
from flyjax.agent.model import base_agent
from flyjax.simulation.simulate import simulate_experiment_jit, simulate_dataset_jit, simulate_dataset_jit_different_params
from flyjax.simulation.parse import parse_reward_matrix
from flyjax.fitting.cv import k_fold_cross_validation_train, \
    k_fold_cross_validation_train_joint, \
    k_fold_cross_validation_train_hierarchical, \
    parallel_k_fold_cross_validation_train, \
    parallel_k_fold_cross_validation_train_joint, \
    parallel_k_fold_cross_validation_train_hierarchical
from flyjax.fitting.joint import evaluate_joint_model
from flyjax.utils.plotting import plot_single_experiment_data
from typing import Tuple, Dict, Any, List, Callable, Optional
from flyjax.fitting.samplers import base_randn_sampler
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
    rng_key, subkey = jax.random.split(rng_key)
    choices, rewards = simulate_dataset_jit(true_params, jnp.stack(reward_matrices), base_agent, subkey, baiting=True)
    n_experiments = len(reward_matrices)
    # convert to experiment data format
    experiments = [(choices[i], rewards[i]) for i in range(n_experiments)]
    
    # Lets generate some experiments with some modified parameters to test the multi group training

    # --- Set perturbed parameters for simulation ---
    # params = [alpha_learn_logit, alpha_forget_logit, kappa_reward, kappa_omission]
    perturbed_params = jnp.array([2.5, 0.0, 0.8, 0.0])
    print("Perturbed parameters:", perturbed_params)

    # --- Simulate a dataset with perturbed parameters ---
    rng_key, subkey = jax.random.split(rng_key)
    perturbed_choices, perturbed_rewards = simulate_dataset_jit(perturbed_params, jnp.stack(reward_matrices), base_agent, subkey, baiting=True)
    n_perturbed_experiments = len(reward_matrices)
    # convert to experiment data format
    perturbed_experiments = [(perturbed_choices[i], perturbed_rewards[i]) for i in range(n_perturbed_experiments)]

    init_theta_sampler = partial(base_randn_sampler, mean=0.0, std=1.0, n_params=4)
    init_delta_sampler = partial(base_randn_sampler, mean=0.0, std=0.1, n_params=4)

    total_pred_ll, per_experiment_ll, params = parallel_k_fold_cross_validation_train_joint(
        experiments_control=experiments,
        experiments_treatment = perturbed_experiments,
        k=5,
        init_theta_sampler=init_theta_sampler,
        init_delta_sampler=init_delta_sampler,
        agent=base_agent,
        learning_rate=5e-2,
        num_steps=10000,
        n_restarts=10,
        min_num_converged=3,
        early_stopping={"min_delta": 1e-4}
    )

    print("Total Predictive Log Likelihood:", total_pred_ll)
    print("Per Experiment Log Likelihoods:", per_experiment_ll)

    theta = np.array(list(params.values())).mean(axis=0)[0]
    delta = np.array(list(params.values())).mean(axis=0)[1]

    # Evaluate the joint model on the two groups.
    nll_control, nll_exp, joint_nll = evaluate_joint_model(
        theta,
        delta,
        base_agent,
        experiments,
        perturbed_experiments,
        delta_penalty_sigma=1.0
    )
    print(f"\nEvaluation:")
    print(f"  Control NLL: {nll_control:.4f}")
    print(f"  Experimental NLL: {nll_exp:.4f}")
    print(f"  Joint NLL: {joint_nll:.4f}")
    print(f"  Delta penalty: {joint_nll - nll_control - nll_exp:.4f}")



