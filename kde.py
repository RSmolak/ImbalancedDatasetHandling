import numpy as np
from scipy.stats import gaussian_kde

def get_kde_weights(data: np.ndarray, transform = None):
    data = data.T
    weights = []

    # Calculate density
    kde = gaussian_kde(data)
    # Calculate weights
    for sample in data.T:
        weights.append(1/kde.evaluate(sample)[0])

    # Transform weights
    if transform == 'normalize':
        weights = [float(i)/sum(weights) for i in weights]
    elif transform == 'standardize':
        weights = np.array(weights)
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        weights = (weights - weight_mean) / weight_std + 1
    elif transform == 'normalize-expand':
        weights = [float(i) / sum(weights) for i in weights]
        mean = np.mean(weights)
        multiplicator  = 1/mean
        weights = [weight * multiplicator for weight in weights]
    elif transform == 'scale_weights_to_mean_one':
        mean_weight = np.mean(weights)
        weights = weights / mean_weight
    return weights