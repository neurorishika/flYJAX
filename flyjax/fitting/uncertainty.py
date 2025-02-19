# flyjax/fitting/uncertainty.py

import jax
import jax.numpy as jnp
import chex
from flyjax.fitting.evaluation import total_negative_log_likelihood

def is_positive_definite(H: jnp.ndarray) -> bool:
    eigenvals = jnp.linalg.eigvalsh(H)
    return jnp.all(eigenvals > 0)

def compute_hessian(loss_fn, params, *args, **kwargs):
    """Compute the Hessian of a scalar-valued loss function at params."""
    H = jax.hessian(loss_fn)(params, *args, **kwargs)
    if not is_positive_definite(H):
        return None
    return H

def laplace_uncertainty(
    params: chex.Array,
    agent: callable,
    experiments: list,
    loss_fn: callable = total_negative_log_likelihood
) -> tuple:
    """
    Estimate uncertainty using a Laplace approximation for a base model.

    It computes the Hessian of the negative log likelihood at the optimum,
    inverts it to get a covariance matrix, and then returns the standard errors.

    Args:
      params: Fitted parameters (a jax array).
      agent: The model function, signature (params, agent_state, choice, reward).
      experiments: List of experiments, where each experiment is a tuple (choices, rewards).
      loss_fn: Function to compute total negative log likelihood (default provided).

    Returns:
      cov: Covariance matrix (inverse Hessian).
      std_errors: Standard errors (sqrt of the diagonal of cov).
    """
    # Define a function that computes the loss for given parameters.
    loss_at_params = lambda p: loss_fn(p, agent, experiments)
    H = compute_hessian(loss_at_params, params)
    if H is None:
        return None, None
    # Invert the Hessian to get the covariance matrix.
    cov = jnp.linalg.inv(H)
    std_errors = jnp.sqrt(jnp.diag(cov))
    return cov, std_errors

def laplace_uncertainty_joint(
    theta: chex.Array,
    delta: chex.Array,
    agent: callable,
    experiments_control: list,
    experiments_treatment: list,
    delta_penalty_sigma: float = 1.0,
    loss_fn: callable = None
) -> tuple:
    """
    Estimate uncertainty for a joint model via a Laplace approximation.

    The joint model defines the experimental parameters as theta_exp = theta + delta.
    The loss function is assumed to be the joint negative log likelihood plus a penalty on delta.

    Args:
      theta: Fitted control group parameters.
      delta: Fitted difference (delta) parameters.
      agent: Joint model function.
      experiments_control: Control group experiments.
      experiments_treatment: Experimental group experiments.
      delta_penalty_sigma: Standard deviation for the quadratic penalty on delta.
      loss_fn: Loss function; if not provided, it uses the one defined in your joint module.

    Returns:
      cov: Covariance matrix estimated via the Laplace approximation.
      std_errors: Standard errors (square roots of the diagonal of cov).
    """
    if loss_fn is None:
        from flyjax.fitting.joint import total_negative_log_likelihood_multi_group
        loss_fn = total_negative_log_likelihood_multi_group

    # Concatenate parameters
    params = jnp.concatenate([theta, delta])
    n_theta = theta.shape[0]

    loss_at_params = lambda p: loss_fn(
        p[:n_theta], p[n_theta:], agent, experiments_control, experiments_treatment, delta_penalty_sigma
    )
    H = compute_hessian(loss_at_params, params)
    if H is None:
        return None, None
    cov = jnp.linalg.inv(H)
    std_errors = jnp.sqrt(jnp.diag(cov))
    return cov, std_errors

def laplace_uncertainty_hierarchical(
    theta_pop: chex.Array,
    theta_subjects: chex.Array,
    agent: callable,
    experiments_by_subject: list,
    sigma_prior: float = 1.0,
    loss_fn: callable = None
) -> tuple:
    """
    Estimate uncertainty for a hierarchical model via Laplace approximation.

    The hierarchical model loss includes a penalty encouraging the subject parameters
    to be close to the population parameter.

    Args:
      theta_pop: Fitted population-level parameters.
      theta_subjects: Fitted subject-specific parameters (2D array: subjects x n_params).
      agent: Hierarchical model function.
      experiments_by_subject: List (one per subject) of experiments.
      sigma_prior: Standard deviation for the Gaussian prior penalty.
      loss_fn: Loss function; if not provided, defaults to total_nll_hierarchical.

    Returns:
      cov: Covariance matrix estimated from the Hessian.
      std_errors: Standard errors (square roots of the diagonal).
    """
    if loss_fn is None:
        from flyjax.fitting.hierarchical import total_nll_hierarchical
        loss_fn = total_nll_hierarchical

    # Flatten subject parameters and concatenate with theta_pop.
    theta_subjects_flat = theta_subjects.flatten()
    params = jnp.concatenate([theta_pop, theta_subjects_flat])
    n_theta = theta_pop.shape[0]

    loss_at_params = lambda p: loss_fn(
        p[:n_theta], p[n_theta:].reshape(theta_subjects.shape), agent, experiments_by_subject, sigma_prior
    )
    H = compute_hessian(loss_at_params, params)
    if H is None:
        return None, None
        
    cov = jnp.linalg.inv(H)
    std_errors = jnp.sqrt(jnp.diag(cov))
    return cov, std_errors
