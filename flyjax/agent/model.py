# agent/model.py
import jax
import jax.numpy as jnp
import chex
from typing import Optional, Tuple

def base_agent(
    params: chex.Array,
    agent_state: Optional[chex.Array] = None,
    choice: Optional[int] = None,
    reward: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cognitive agent for a binary two-armed bandit task.
    Maintains Q-values and returns a softmax policy.
    If a choice and reward are provided, the agent updates its Q-values.
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
