# flyjax/agent/model_zoo_rl.py
import jax
import jax.numpy as jnp
import chex
from typing import Optional, Tuple, Dict

def q_agent(
    params: chex.Array,
    agent_state: Optional[chex.Array] = None,
    choice: Optional[int] = None,
    reward: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Standard RL agent using Q-learning.
    
    Parameters:
      - params: An array of 2 values: [alpha_learn_logit, kappa_reward].
      - agent_state: The current Q-values. If None, initializes to [0.5, 0.5].
      - choice, reward: If provided, the agent updates its Q-values.

    Returns:
        - probs: The softmax-transformed Q-values (policy).
        - new_agent_state: The updated Q-values.
    """
    if agent_state is None:
        agent_state = jnp.array([0.5, 0.5])
    probs = jax.nn.softmax(agent_state)
    if choice is None or reward is None:
        return probs, agent_state

    alpha_learn = 1 / (1 + jnp.exp(-params[0]))
    kappa_reward = params[1]
    learn_target = kappa_reward * reward
    qs = agent_state
    qs = qs.at[choice].set(qs[choice] + alpha_learn * (learn_target - qs[choice]))
    new_probs = jax.nn.softmax(qs)
    return new_probs, qs

def forgetting_q_agent(
    params: chex.Array,
    agent_state: Optional[chex.Array] = None,
    choice: Optional[int] = None,
    reward: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Standard RL agent using Q-learning with forgetting.

    Parameters:
        - params: An array of 3 values: [alpha_learn_logit, alpha_forget_logit, kappa_reward].
        - agent_state: The current Q-values. If None, initializes to [0.5, 0.5].
        - choice, reward: If provided, the agent updates its Q-values.

    Returns:
        - probs: The softmax-transformed Q-values (policy).
        - new_agent_state: The updated Q-values.
    """
    if agent_state is None:
        agent_state = jnp.array([0.5, 0.5])
    probs = jax.nn.softmax(agent_state)
    if choice is None or reward is None:
        return probs, agent_state

    alpha_learn = 1 / (1 + jnp.exp(-params[0]))
    alpha_forget = 1 / (1 + jnp.exp(-params[1]))
    kappa_reward = params[2]
    learn_target = kappa_reward * reward
    qs = agent_state
    qs = qs.at[choice].set(qs[choice] + alpha_learn * (learn_target - qs[choice]))
    qs = qs.at[1 - choice].set((1 - alpha_forget) * qs[1 - choice])
    new_probs = jax.nn.softmax(qs)
    return new_probs, qs


def differential_forgetting_q_agent(
    params: chex.Array,
    agent_state: Optional[chex.Array] = None,
    choice: Optional[int] = None,
    reward: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Standard RL agent using differential forgetting Q-learning.
    
    Parameters:
      - params: An array of 4 values: [alpha_learn_logit, alpha_forget_logit, 
                kappa_reward, kappa_omission].
      - agent_state: The current Q-values. If None, initializes to [0.5, 0.5].
      - choice, reward: If provided, the agent updates its Q-values.
      
    Returns:
      - probs: The softmax-transformed Q-values (policy).
      - new_agent_state: The updated Q-values.
    """
    if agent_state is None:
        agent_state = jnp.array([0.5, 0.5])
    probs = jax.nn.softmax(agent_state)
    if choice is None or reward is None:
        return probs, agent_state

    alpha_learn = 1 / (1 + jnp.exp(-params[0]))
    alpha_forget = 1 / (1 + jnp.exp(-params[1]))
    kappa_reward = params[2]
    kappa_omission = params[3]
    learn_target = kappa_reward * reward + kappa_omission * (1 - reward)
    qs = agent_state
    qs = qs.at[choice].set(qs[choice] + alpha_learn * (learn_target - qs[choice]))
    qs = qs.at[1 - choice].set((1 - alpha_forget) * qs[1 - choice])
    new_probs = jax.nn.softmax(qs)
    return new_probs, qs

def advanced_rl_agent(
    params: chex.Array,
    agent_state: Optional[chex.Array] = None,
    choice: Optional[int] = None,
    reward: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    RL agent that combines multiple features:
      - Differential forgetting Q-learning.
      - Learnable initial Q-values.
      - Dual learning rates for positive and negative prediction errors.
      - Epsilon-greedy exploration with a softmax policy.
      
    Parameters:
      - params: An array of 9 values:
          [q0_init, q1_init, alpha_pos_logit, alpha_neg_logit, alpha_forget_logit, 
           kappa_reward, kappa_omission, epsilon, temperature].
             p[0:2] -> initial Q-values,
             p[2] -> alpha_pos_logit,
             p[3] -> alpha_neg_logit,
             p[4] -> alpha_forget_logit,
             p[5] -> kappa_reward,
             p[6] -> kappa_omission,
             p[7] -> epsilon,
             p[8] -> temperature
      - agent_state: The current Q-values. If None, initialize using the first two parameters.
      - choice, reward: If provided, update Q-values.
      
    Returns:
      - probs: The probability distribution over actions.
      - new_agent_state: Updated Q-values.
    """
    # For convenience, assume params is of length 10:
    # [q0_init, q1_init, alpha_pos_logit, alpha_neg_logit, alpha_forget_logit, kappa_reward, kappa_omission, epsilon]
    if agent_state is None:
        agent_state = jnp.array([params[0], params[1]])
    
    # Compute softmax with temperature.
    softmax_probs = jax.nn.softmax(agent_state)
    epsilon = params[7]
    probs = (1 - epsilon) * softmax_probs + epsilon * jnp.array([0.5, 0.5])
    
    if choice is None or reward is None:
        return probs, agent_state
    
    alpha_pos = 1 / (1 + jnp.exp(-params[2]))
    alpha_neg = 1 / (1 + jnp.exp(-params[3]))
    alpha_forget = 1 / (1 + jnp.exp(-params[4]))
    kappa_reward = params[5]
    kappa_omission = params[6]
    
    current_q = agent_state[choice]
    learn_target = kappa_reward * reward + kappa_omission * (1 - reward)
    error = learn_target - current_q
    alpha_learn = jnp.where(error >= 0, alpha_pos, alpha_neg)
    
    qs = agent_state
    qs = qs.at[choice].set(current_q + alpha_learn * error)
    qs = qs.at[1 - choice].set((1 - alpha_forget) * qs[1 - choice])
    new_probs = jax.nn.softmax(qs / temperature)
    new_probs = (1 - epsilon) * new_probs + epsilon * jnp.array([0.5, 0.5])
    return new_probs, qs

