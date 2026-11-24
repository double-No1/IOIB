"""Weighted 1-D two-Gaussian mixture and crossover threshold.

Samples are weighted by
``1 - anomaly_score`` so the fit is biased toward nominal predictions
(high precision, lower recall of nominal images).
"""
import math

import numpy as np


def find_gaussian_crossover(mu1, sigma1, w1, mu2, sigma2, w2):
    a = 1 / (2 * sigma1**2) - 1 / (2 * sigma2**2)
    b = mu2 / (sigma2**2) - mu1 / (sigma1**2)
    c = (mu1**2) / (2 * sigma1**2) - (mu2**2) / (2 * sigma2**2) + np.log(
        (w2 * sigma1) / (w1 * sigma2)
    )
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return []
    if np.isclose(discriminant, 0):
        return [-b / (2 * a)]
    sqrt_disc = np.sqrt(discriminant)
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)
    return sorted([x1, x2])


def weighted_gmm_em_1d_crossover(x, w, n_iters=100, tol=1e-6, verbose=False):
    n_components = 2
    n = x.shape[0]
    x = x.reshape(-1, 1)
    w = w / np.sum(w)

    means = np.random.choice(x.flatten(), n_components)
    stds = np.full(n_components, np.std(x))
    weights = np.full(n_components, 1 / n_components)
    log_likelihood_old = -np.inf

    for _ in range(n_iters):
        responsibilities = np.zeros((n, n_components))
        for k in range(n_components):
            coeff = weights[k] / (np.sqrt(2 * np.pi) * stds[k])
            exp_term = np.exp(-0.5 * ((x.flatten() - means[k]) / stds[k]) ** 2)
            responsibilities[:, k] = coeff * exp_term
        responsibilities = responsibilities / np.sum(
            responsibilities, axis=1, keepdims=True
        )
        responsibilities *= w[:, np.newaxis]
        Nk = np.sum(responsibilities, axis=0)
        for k in range(n_components):
            means[k] = np.sum(responsibilities[:, k] * x.flatten()) / Nk[k]
            var_k = np.sum(responsibilities[:, k] * (x.flatten() - means[k]) ** 2) / Nk[k]
            stds[k] = np.sqrt(var_k)
            weights[k] = Nk[k]
        weights /= np.sum(weights)

        log_probs = np.zeros((n, n_components))
        for k in range(n_components):
            coeff = weights[k] / (np.sqrt(2 * np.pi) * stds[k])
            exp_term = np.exp(-0.5 * ((x.flatten() - means[k]) / stds[k]) ** 2)
            log_probs[:, k] = coeff * exp_term
        log_likelihood = np.sum(w * np.log(np.sum(log_probs, axis=1) + 1e-12))
        if np.abs(log_likelihood - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood

    if verbose:
        print(means[0], stds[0], weights[0], "    ", means[1], stds[1], weights[1])

    crossover_points = find_gaussian_crossover(
        mu1=means[0],
        sigma1=stds[0],
        w1=weights[0],
        mu2=means[1],
        sigma2=stds[1],
        w2=weights[1],
    )
    gaussian_0 = {"mu": means[0], "sigma": stds[0], "weight": weights[0]}
    gaussian_1 = {"mu": means[1], "sigma": stds[1], "weight": weights[1]}
    if verbose:
        print(crossover_points)
    crossover_points = np.array(crossover_points)
    if len(crossover_points) > 0:
        return (
            crossover_points[np.argmin((crossover_points - 0.45) ** 2)],
            gaussian_0,
            gaussian_1,
        )
    return np.nan, gaussian_0, gaussian_1


def dual_gaussian_thresh(flat_preds, plot_name=None, verbose=False):
    values, nom, anom = [], [], []
    for key in sorted(flat_preds):
        values.append(flat_preds[key])
        if "_" in key.split("/")[-1]:
            anom.append(flat_preds[key])
        else:
            nom.append(flat_preds[key])
    values = np.array(values)

    threshold, g_0, g_1 = weighted_gmm_em_1d_crossover(
        values, 1.0 - values, verbose=verbose
    )
    attempts = 0
    while math.isnan(threshold):
        threshold, g_0, g_1 = weighted_gmm_em_1d_crossover(
            values, 1.0 - values, verbose=verbose
        )
        attempts += 1
        if attempts > 5:
            threshold = np.mean(values) + np.std(values)

    if plot_name:
        from baaf.viz import plot_two_histograms

        plot_two_histograms(
            nom, anom, plot_name=plot_name, vline=threshold, gaussian_0=g_0, gaussian_1=g_1
        )
    return threshold
