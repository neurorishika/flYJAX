# simulation/simulate.py
import jax
import jax.numpy as jnp
from functools import partial
import chex
from typing import Tuple, List, Callable
from tqdm import tqdm

def simulate_experiment(
    params: chex.Array,
    reward_matrix: jnp.ndarray,
    agent: Callable,
    rng_key: chex.Array,
    baiting: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate a single experiment using the agent, with an optional baiting mechanism.
    
    Args:
        params: The true parameters used to simulate the agent.
        reward_matrix: A (num_trials, 2) array, where each row specifies the reward
                       probability for each option on that trial.
        agent: The agent function, which should have the signature:
                (params, agent_state, choice, reward) -> (new_probs, new_state)
        rng_key: A JAX random key.
        baiting: If True, implements baiting. In baiting mode, at the beginning of each trial,
                 each option is sampled for a reward with the corresponding probability
                 (if not already baited). Once an option is baited, it remains baited until the agent 
                 chooses it, at which point the reward is provided and the bait is consumed.
                 
    Returns:
        choices: Array of simulated choices.
        rewards: Array of simulated rewards (only provided when the bait is collected).
    """
    num_trials = reward_matrix.shape[0]
    agent_state = jnp.array([0.5, 0.5])
    choices_list = []
    rewards_list = []
    
    # In baiting mode, maintain a state that indicates whether each option is baited.
    if baiting:
        # Use booleans: False means not baited, True means baited.
        bait_state = jnp.array([False, False])
    
    for t in range(num_trials):
        # If baiting is enabled, update the bait state for both options.
        if baiting:
            # For each option, if it is not already baited, sample a reward.
            new_bait_state = list(bait_state)  # temporary mutable copy
            for option in [0, 1]:
                if not bait_state[option]:
                    rng_key, subkey = jax.random.split(rng_key)
                    # Sample whether the option becomes baited on this trial.
                    sample = bool(jax.random.bernoulli(subkey, p=reward_matrix[t, option]))
                    new_bait_state[option] = sample
            bait_state = jnp.array(new_bait_state)
        
        # Retrieve the current policy from the agent.
        cur_probs, _ = agent(params, agent_state)
        rng_key, subkey = jax.random.split(rng_key)
        choice = int(jax.random.choice(subkey, a=jnp.array([0, 1]), p=cur_probs))
        
        if baiting:
            # In baiting mode, if the chosen option is baited, deliver a reward and clear the bait.
            if bait_state[choice]:
                reward = 1
                bait_state = bait_state.at[choice].set(False)  # consume the bait
            else:
                reward = 0
        else:
            # In non-baiting mode, sample reward only for the chosen option on this trial.
            rng_key, subkey = jax.random.split(rng_key)
            reward = int(jax.random.bernoulli(subkey, p=reward_matrix[t, choice]))
        
        choices_list.append(choice)
        rewards_list.append(reward)
        
        # Update the agent state using the observed outcome.
        _, agent_state = agent(params, agent_state, choice, reward)
    
    return jnp.array(choices_list), jnp.array(rewards_list)

def simulate_dataset(
    params: chex.Array,
    reward_matrices: List[jnp.ndarray],
    agent: Callable,
    rng_key: chex.Array,
    baiting: bool = False
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Simulate a dataset given a list of reward matrices.
    
    Each reward matrix should be a (num_trials, 2) array specifying the reward
    probabilities for each option on each trial.
    
    Args:
        params: The true parameters used for simulation.
        reward_matrices: A list of reward matrices (one per experiment).
        agent: The agent function, which should have the signature:
                (params, agent_state, choice, reward) -> (new_probs, new_state)
        rng_key: A JAX random key.
        baiting: If True, the simulation uses the baiting mechanism described above.
        
    Returns:
        A list where each element is a tuple (choices, rewards) for one experiment.
    """
    experiments = []
    for reward_matrix in tqdm(reward_matrices, desc="Simulating experiments"):
        rng_key, subkey = jax.random.split(rng_key)
        exp_data = simulate_experiment(params, reward_matrix, agent, subkey, baiting=baiting)
        experiments.append(exp_data)
    return experiments

def simulate_dataset_different_params(
    params_stack: List[chex.Array],
    reward_matrices: List[jnp.ndarray],
    agent: Callable,
    rng_key: chex.Array,
    baiting: bool = False
) -> List[List[Tuple[jnp.ndarray, jnp.ndarray]]]:
    """
    Simulate a dataset given a list of reward matrices and a list of parameter sets.

    Each reward matrix should be a (num_trials, 2) array specifying the reward
    probabilities for each option on each trial.

    Args:
        params_stack: A list of parameter sets (one per experiment).
        reward_matrices: A list of reward matrices (one per experiment).
        agent: The agent function, which should have the signature:
                (params, agent_state, choice, reward) -> (new_probs, new_state)
        rng_key: A JAX random key.
        baiting: If True, the simulation uses the baiting mechanism described above.

    Returns:
        A list where each element is a list of tuples (choices, rewards) for one experiment.
    """
    experiments = []
    for params, reward_matrix in tqdm(zip(params_stack, reward_matrices), desc="Simulating datasets"):
        exp_data = simulate_dataset(params, reward_matrix, agent, rng_key, baiting=baiting)
        experiments.append(exp_data)
    return experiments


@partial(jax.jit, static_argnames=['agent','baiting'])
def simulate_experiment_jit(
        params: chex.Array,
        reward_matrix: jnp.ndarray,
        agent: Callable,
        rng_key: chex.Array,
        baiting: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate a single experiment using a JIT-compiled loop.
    
    Args:
        params: Agent parameters.
        reward_matrix: A (num_trials, 2) array of reward probabilities.
        agent: The agent function.
        rng_key: A JAX random key.
        baiting: (bool) Whether to use the baiting mechanism.
    
    Returns:
        choices: An array of choices for each trial.
        rewards: An array of rewards for each trial.
    """
    num_trials = reward_matrix.shape[0]
    # Initialize agent state to [0.5, 0.5]
    agent_state = jnp.array([0.5, 0.5])
    # For baiting, we maintain an extra state: a boolean array for each option.
    init_bait_state = jnp.array([False, False]) if baiting else None

    # Define the per-trial update function.
    def trial_step(carry, t):
        # Depending on baiting, our carry is a tuple:
        #   (agent_state, rng_key, bait_state)  if baiting==True,
        #   (agent_state, rng_key)              if baiting==False.
        if baiting:
            agent_state, rng_key, bait_state = carry
        else:
            agent_state, rng_key = carry

        # === If baiting is enabled, update bait state for both options ===
        if baiting:
            # For each option, if it is not already baited, sample a new bait.
            # We perform the update for option 0:
            rng_key, subkey0 = jax.random.split(rng_key)
            bait_state_0 = jax.lax.cond(
                bait_state[0],
                lambda _: True,  # already baited → leave as True
                lambda _: jax.random.bernoulli(subkey0, p=reward_matrix[t, 0]),
                operand=None,
            )
            # And for option 1:
            rng_key, subkey1 = jax.random.split(rng_key)
            bait_state_1 = jax.lax.cond(
                bait_state[1],
                lambda _: True,
                lambda _: jax.random.bernoulli(subkey1, p=reward_matrix[t, 1]),
                operand=None,
            )
            bait_state = jnp.array([bait_state_0, bait_state_1])
        
        # === Agent makes a choice ===
        # Get the current policy (probabilities) from the agent.
        cur_probs, _ = agent(params, agent_state)
        rng_key, subkey = jax.random.split(rng_key)
        choice = jax.random.choice(subkey, a=jnp.array([0, 1]), p=cur_probs)
        
        # === Determine reward based on baiting flag ===
        if baiting:
            # If the chosen option is baited, the agent receives a reward (and the bait is consumed).
            reward = jnp.where(bait_state[choice], 1, 0)
            # Clear the bait if it was collected.
            bait_state = jax.lax.cond(
                bait_state[choice],
                lambda _: bait_state.at[choice].set(False),
                lambda _: bait_state,
                operand=None,
            )
        else:
            # No baiting: sample reward for the chosen option.
            rng_key, subkey = jax.random.split(rng_key)
            reward = jax.random.bernoulli(subkey, p=reward_matrix[t, choice]).astype(jnp.int32)
        
        # === Update the agent state using the observed outcome ===
        _, agent_state = agent(params, agent_state, choice, reward)
        
        # Repack the carry.
        if baiting:
            new_carry = (agent_state, rng_key, bait_state)
        else:
            new_carry = (agent_state, rng_key)
        # Return the trial’s choice and reward.
        return new_carry, (choice, reward)
    
    # Set the initial carry.
    if baiting:
        init_carry = (agent_state, rng_key, init_bait_state)
    else:
        init_carry = (agent_state, rng_key)
    
    # Run the scan over trials.
    carry, (choices, rewards) = jax.lax.scan(trial_step, init_carry, jnp.arange(num_trials))
    return choices, rewards


@partial(jax.jit, static_argnames=['agent','baiting'])
def simulate_dataset_jit(
    params: chex.Array,
    reward_matrix_stack: jnp.ndarray,
    agent: Callable,
    rng_key: chex.Array,
    baiting: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorize simulation over multiple experiments.
    
    Args:
        params: Agent parameters.
        reward_matrix_stack: A stack of reward matrices of shape (n_experiments, num_trials, 2).    
        agent: The agent function.
        rng_key: A base random key.
        baiting: Whether to use baiting.
    
    Returns:
        choices: Array of shape (n_experiments, num_trials)
        rewards: Array of shape (n_experiments, num_trials)
    """
    n_experiments = reward_matrix_stack.shape[0]
    # Split the base rng_key into one per experiment.
    keys = jax.random.split(rng_key, n_experiments)
    # Use vmap to apply simulate_experiment_jit to each experiment.
    sim_fn = lambda key, rew: simulate_experiment_jit(params, rew, agent, key, baiting)
    choices, rewards = jax.vmap(sim_fn)(keys, reward_matrix_stack)
    return choices, rewards


@partial(jax.jit, static_argnames=['agent','baiting'])
def simulate_dataset_jit_different_params(
    params_stack: chex.Array,
    reward_matrix_stack: jnp.ndarray,
    agent: Callable,
    rng_key: chex.Array,
    baiting: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized simulation over multiple experiments with different parameters.
    
    Args:
        params_stack: A stack of parameter sets of shape (n_experiments, n_params).
        reward_matrix_stack: A stack of reward matrices of shape (n_experiments, num_trials, 2).
        rng_key: A base random key.
        baiting: Whether to use baiting.
    
    Returns:
        choices: Array of shape (n_experiments, num_trials)
        rewards: Array of shape (n_experiments, num_trials)
    """
    n_experiments = reward_matrix_stack.shape[0]
    # Split the base rng_key into one per experiment.
    keys = jax.random.split(rng_key, n_experiments)
    # Use vmap to apply simulate_experiment_jit to each experiment.
    sim_fn = lambda key, params, rew: simulate_experiment_jit(params, rew, agent, key, baiting)
    choices, rewards = jax.vmap(sim_fn, in_axes=(0, 0, 0))(keys, params_stack, reward_matrix_stack)
    return choices, rewards

