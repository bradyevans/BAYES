import pandas as pd
import numpy as np

def beta_mean(a, b):
    return a / (a + b)

def beta_mode(a, b):
    if a < 1 and b < 1:
        mode = "0 and 1"
    elif a <= 1 and b > 1:
        mode = 0
    elif a > 1 and b <= 1:
        mode = 1
    else:
        mode = (a - 1) / (a + b - 2)
    return mode

def beta_var(a, b):
    return a * b / ((a + b) ** 2 * (a + b + 1))

def summarize_beta_binomial(alpha, beta, y=None, n=None):
    prior_mean = beta_mean(alpha, beta)
    prior_mode = beta_mode(alpha, beta)
    prior_var = beta_var(alpha, beta)
    prior_sd = np.sqrt(prior_var)
    
    summary = {
        'model': ['prior'],
        'alpha': [alpha],
        'beta': [beta],
        'mean': [prior_mean],
        'mode': [prior_mode],
        'var': [prior_var],
        'sd': [prior_sd]
    }
    
    if y is not None and n is not None:
        post_alpha = y + alpha
        post_beta = n - y + beta
        post_mean = beta_mean(post_alpha, post_beta)
        post_mode = beta_mode(post_alpha, post_beta)
        post_var = beta_var(post_alpha, post_beta)
        post_sd = np.sqrt(post_var)
        
        summary['model'].append('posterior')
        summary['alpha'].append(post_alpha)
        summary['beta'].append(post_beta)
        summary['mean'].append(post_mean)
        summary['mode'].append(post_mode)
        summary['var'].append(post_var)
        summary['sd'].append(post_sd)
    
    return pd.DataFrame(summary)