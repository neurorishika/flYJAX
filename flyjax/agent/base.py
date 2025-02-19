# agent/model.py
import jax
import jax.numpy as jnp
import chex
from typing import Optional, Tuple

def random_policy(
    params: chex.Array,
    agent_state: Optional[chex.Array] = None,
    choice: Optional[int] = None,
    reward: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cognitive agent for a binary two-armed bandit task with a random policy.
    """
    return jnp.array([0.5, 0.5]), jnp.array([0.5, 0.5])

def learned_fixed_policy(
    params: chex.Array,
    agent_state: Optional[chex.Array] = None,
    choice: Optional[int] = None,
    reward: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cognitive agent for a binary two-armed bandit task with a fixed learned policy.
    """
    if agent_state is None:
        agent_state = jnp.array([params[0], params[1]])
    probs = jax.nn.softmax(agent_state)
    return probs, agent_state

def one_bit_agent(
    params: chex.Array,
    agent_state: Optional[chex.Array] = None,
    choice: Optional[int] = None,
    reward: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cognitive agent for a binary two-armed bandit task with a one-bit memory to make win-stay, lose-shift decisions.

    The agent has two parameters:
    - params[0]: win-stay probability
    - params[1]: lose-shift probability

    The agent maintains a state = (last_choice, last_reward).
    """
    win_stay = 1 / (1 + jnp.exp(-params[0]))
    lose_shift = 1 / (1 + jnp.exp(-params[1]))

    if agent_state is None:
        agent_state = jnp.array([-1, -1])

    if choice is None or reward is None:
        # calculate probabilities based on last choice and reward
        last_choice, last_reward = agent_state
        if last_choice == 0 and last_reward == 1:
            return jnp.array([win_stay, 1 - win_stay]), agent_state
        elif last_choice == 1 and last_reward == 1:
            return jnp.array([1 - win_stay, win_stay]), agent_state
        elif last_choice == 0 and last_reward == 0:
            return jnp.array([1 - lose_shift, lose_shift]), agent_state
        elif last_choice == 1 and last_reward == 0:
            return jnp.array([lose_shift, 1 - lose_shift]), agent_state
        else:
            return jnp.array([0.5, 0.5]), agent_state

    # update state based on current choice and reward
    new_state = jnp.array([choice, reward])
    # recalculate probabilities based on current choice and reward
    if agent_state[0] == 0 and agent_state[1] == 1:
        return jnp.array([win_stay, 1 - win_stay]), new_state
    elif agent_state[0] == 1 and agent_state[1] == 1:
        return jnp.array([1 - win_stay, win_stay]), new_state
    elif agent_state[0] == 0 and agent_state[1] == 0:
        return jnp.array([1 - lose_shift, lose_shift]), new_state
    elif agent_state[0] == 1 and agent_state[1] == 0:
        return jnp.array([lose_shift, 1 - lose_shift]), new_state
    else:
        return jnp.array([0.5, 0.5]), new_state

def test_agent(
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
