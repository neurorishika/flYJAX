from scipy.stats import chi2


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
    p_value = 1 - chi2.cdf(test_stat, df)
    return p_value