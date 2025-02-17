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