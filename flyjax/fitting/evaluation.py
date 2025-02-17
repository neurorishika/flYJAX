import chex
import jax
import jax.numpy as jnp
from typing import Callable, List, Tuple

def log_likelihood_experiment(
    params: chex.Array,
    agent: Callable,
    choices: chex.Array,
    rewards: chex.Array
) -> jnp.ndarray:
    """
    Compute the log likelihood for one experiment given the observed choices
    and rewards.
    
    At each trial, the current agent policy (probabilities) is used to compute
    the log likelihood of the observed choice. Then the agent is updated based on
    the choice and reward.
    """
    init_state = jnp.array([0.5, 0.5])
    
    def trial_step(agent_state: chex.Array, trial_data: jnp.ndarray):
        # trial_data: [choice, reward]
        choice, reward = trial_data.astype(jnp.int32)
        # Retrieve current policy from the agent.
        cur_probs, _ = agent(params, agent_state)
        log_prob = jnp.log(cur_probs[choice] + 1e-8)
        # Update agent state given the trial outcome.
        _, new_state = agent(params, agent_state, choice, reward)
        return new_state, log_prob
    
    trial_data = jnp.stack([choices, rewards], axis=1)
    _, log_probs = jax.lax.scan(trial_step, init_state, trial_data)
    return jnp.sum(log_probs)

def negative_log_likelihood_experiment(
    params: chex.Array,
    agent: Callable,
    choices: chex.Array,
    rewards: chex.Array
) -> jnp.ndarray:
    """Returns the negative log likelihood for a single experiment."""
    return -log_likelihood_experiment(params, agent, choices, rewards)

def total_negative_log_likelihood(
    params: chex.Array,
    agent: Callable,
    experiments: List[Tuple[chex.Array, chex.Array]]
) -> jnp.ndarray:
    """
    Sum the negative log likelihoods over multiple experiments.
    """
    total_nll = 0.0
    for choices, rewards in experiments:
        total_nll += negative_log_likelihood_experiment(params, agent, choices, rewards)
    return total_nll

def aic(log_likelihood: float, num_params: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).
    
    AIC = 2 * k - 2 * log(L)
    
    Args:
        log_likelihood: The log-likelihood of the model.
        num_params: The number of free parameters in the model.
        
    Returns:
        AIC value.
    """
    return 2 * num_params - 2 * log_likelihood

def bic(log_likelihood: float, num_params: int, num_data: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).
    
    BIC = k * ln(n) - 2 * log(L)
    
    Args:
        log_likelihood: The log-likelihood of the model.
        num_params: The number of free parameters in the model.
        num_data: The number of observations (e.g., total number of trials).
        
    Returns:
        BIC value.
    """
    return num_params * jnp.log(num_data) - 2 * log_likelihood

def likelihood_ratio_test(
    ll_full: float,
    ll_restricted: float,
    num_params_full: int,
    num_params_restricted: int
) -> float:
    """
    Perform a likelihood ratio test comparing a full model to a nested restricted model.
    
    Args:
        ll_full: Log-likelihood of the full model.
        ll_restricted: Log-likelihood of the restricted (nested) model.
        num_params_full: Number of free parameters in the full model.
        num_params_restricted: Number of free parameters in the restricted model.
        
    Returns:
        p-value of the test, assuming the test statistic follows a chi-square distribution.
    """
    # The test statistic is 2 * (ll_full - ll_restricted)
    test_stat = 2 * (ll_full - ll_restricted)
    df = num_params_full - num_params_restricted
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(test_stat, df)
    return p_value