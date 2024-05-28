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
    elif transform == 'scale_weights_to_mean_one_squared':
        mean_weight = np.mean(weights)
        weights = weights / mean_weight
        weights = [weight**2 for weight in weights]
    return weights


def generate_kde_samples(data: np.ndarray, num_samples: int):
    data = data.T  # Transpose data to match the expected shape by gaussian_kde (features in rows)

    try:
        kde = gaussian_kde(data)  # Attempt to fit the KDE model
    except np.linalg.LinAlgError:
        # Apply a stronger regularization and retry
        kde = gaussian_kde(data, bw_method='scott')
        kde.covariance = kde.covariance + 0.0001 * np.eye(data.shape[1])
        kde._compute_covariance()

    # Resample new points from the KDE model
    new_samples = kde.resample(size=num_samples).T  # Transpose back to original shape (samples, features)
    return new_samples
