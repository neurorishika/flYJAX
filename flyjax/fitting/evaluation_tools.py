import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Callable
import chex
from .evaluation import log_likelihood_experiment, total_negative_log_likelihood

# -----------------------------------------------------------------------------
# Information Criteria
# -----------------------------------------------------------------------------
def compute_aic(log_likelihood: float, num_params: int) -> float:
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

def compute_bic(log_likelihood: float, num_params: int, num_data: int) -> float:
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
    return num_params * np.log(num_data) - 2 * log_likelihood

def compute_waic(log_likelihoods: jnp.ndarray) -> Tuple[float, float]:
    """
    Compute WAIC (Widely Applicable Information Criterion) given an array
    of log-likelihoods for each data point and posterior sample.
    
    Here, we assume `log_likelihoods` is a 2D array of shape (n_samples, n_data)
    where each row corresponds to a posterior sample and each column to an observation.
    
    Returns:
        A tuple (waic, p_waic) where p_waic is the effective number of parameters.
    
    Note: In practice, you might use libraries like ArviZ for WAIC computation.
    """
    # Compute the log pointwise predictive density (lppd)
    lppd = jnp.sum(jax.scipy.special.logsumexp(log_likelihoods, axis=0) - jnp.log(log_likelihoods.shape[0]))
    # Compute the variance across posterior samples for each data point.
    p_waic = jnp.sum(jnp.var(log_likelihoods, axis=0))
    waic = -2 * (lppd - p_waic)
    return float(waic), float(p_waic)

# -----------------------------------------------------------------------------
# K-Fold Cross-Validation
# -----------------------------------------------------------------------------
def k_fold_split(experiments: List[Tuple[chex.Array, chex.Array]], k: int) -> List[Tuple[List, List]]:
    """
    Split the experiments into k folds.
    
    Returns a list of tuples (train_set, test_set) for each fold.
    """
    n = len(experiments)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    splits = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        train_set = [experiments[idx] for idx in train_idx]
        test_set = [experiments[idx] for idx in test_idx]
        splits.append((train_set, test_set))
    return splits

def cross_validation_score(
    model_fn: Callable[[chex.Array, Tuple[chex.Array, chex.Array]], Tuple[jnp.ndarray, chex.Array]],
    init_params: chex.Array,
    experiments: List[Tuple[chex.Array, chex.Array]],
    k: int = 3,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    train_fn: Callable = None,
) -> float:
    """
    Evaluate a model's predictive performance via k-fold cross-validation.
    
    Args:
        model_fn: A function that, given parameters and an experiment (choices, rewards),
                  returns predictions or computes likelihood.
        init_params: Initial parameters for training.
        experiments: List of experiments.
        k: Number of folds.
        learning_rate: Learning rate for model training.
        num_steps: Number of training steps.
        train_fn: A training function that fits parameters on a training set.
                  If None, use your default train_model.
    
    Returns:
        The average negative log likelihood (or predictive error) on the held-out data.
    """
    # Use your default train_model if not provided.
    if train_fn is None:
        from .train import train_model  # import default training routine
        train_fn = train_model

    splits = k_fold_split(experiments, k)
    cv_scores = []
    
    for train_set, test_set in splits:
        # Train the model on the training set.
        trained_params = train_fn(init_params, train_set, learning_rate=learning_rate, num_steps=num_steps)
        # Evaluate the model on the test set.
        test_nll = total_negative_log_likelihood(trained_params, test_set)
        cv_scores.append(test_nll)
    return float(np.mean(cv_scores))

# -----------------------------------------------------------------------------
# Likelihood Ratio Test for Nested Models
# -----------------------------------------------------------------------------
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
