"""Streaming Bayesian update engine for real-time psychometric estimation.

Provides sequential Monte Carlo (SMC) and variational inference (VI)
backends so that posterior estimates of threshold and slope update
incrementally as new trial observations arrive.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from ..types import InferenceMethod, StreamingConfig, StreamingState


def initialize(config: StreamingConfig) -> StreamingState:
    """Create an initial streaming state from *config*.

    For SMC the particles are drawn from the prior; for VI the
    variational parameters are set to prior moments.
    """
    rng = np.random.default_rng(config.random_seed)

    if config.method in (InferenceMethod.SMC, InferenceMethod.MCMC):
        thresholds = rng.normal(0.0, 1.0, size=config.n_particles)
        slopes = np.abs(rng.normal(0.0, 2.0, size=config.n_particles))
        particles = np.column_stack([thresholds, slopes])
        log_weights = np.zeros(config.n_particles)
        return StreamingState(
            method=config.method,
            step=0,
            particles=particles,
            log_weights=log_weights,
            ess_history=[float(config.n_particles)],
        )

    # Variational inference initialization
    return StreamingState(
        method=config.method,
        step=0,
        vi_means=np.array([0.0, 1.0]),
        vi_stds=np.array([1.0, 1.0]),
        ess_history=[],
    )


def update(
    state: StreamingState,
    intensity: float,
    response: int,
    config: StreamingConfig,
) -> StreamingState:
    """Incorporate a single trial observation into the streaming posterior.

    Parameters
    ----------
    state:
        Current estimator state.
    intensity:
        Stimulus intensity for this trial.
    response:
        Binary outcome (1 = detected / hit, 0 = miss).
    config:
        Streaming configuration (controls resampling threshold, etc.).

    Returns
    -------
    Updated :class:`StreamingState` reflecting the new observation.
    """
    if state.method in (InferenceMethod.SMC, InferenceMethod.MCMC):
        return _smc_update(state, intensity, response, config)
    return _vi_update(state, intensity, response, config)


# ---------------------------------------------------------------------------
# Sequential Monte Carlo helpers
# ---------------------------------------------------------------------------


def _log_likelihood(
    intensity: float,
    response: int,
    particles: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Bernoulli log-likelihood for each particle given a single trial."""
    thresholds = particles[:, 0]
    slopes = particles[:, 1]
    p = expit(slopes * (intensity - thresholds))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    if response == 1:
        return np.log(p)
    return np.log1p(-p)


def _smc_update(
    state: StreamingState,
    intensity: float,
    response: int,
    config: StreamingConfig,
) -> StreamingState:
    """Weight → resample cycle for one observation."""
    ll = _log_likelihood(intensity, response, state.particles)
    new_log_weights = state.log_weights + ll

    # Normalize and compute ESS
    shifted = new_log_weights - np.max(new_log_weights)
    weights = np.exp(shifted)
    weights /= weights.sum()
    ess = float(1.0 / np.sum(weights**2))

    particles = state.particles
    log_weights = new_log_weights

    # Resample when ESS drops below threshold
    if ess < config.ess_threshold * config.n_particles:
        rng = np.random.default_rng(
            None if config.random_seed is None else config.random_seed + state.step,
        )
        indices = rng.choice(
            len(particles),
            size=config.n_particles,
            p=weights,
        )
        particles = particles[indices]
        # Jitter to avoid particle collapse
        particles = particles + rng.normal(0, 0.01, size=particles.shape)
        particles[:, 1] = np.abs(particles[:, 1])  # slopes stay positive
        log_weights = np.zeros(config.n_particles)
        ess = float(config.n_particles)

    return StreamingState(
        method=state.method,
        step=state.step + 1,
        particles=particles,
        log_weights=log_weights,
        ess_history=[*state.ess_history, ess],
    )


# ---------------------------------------------------------------------------
# Variational inference helpers
# ---------------------------------------------------------------------------


def _vi_update(
    state: StreamingState,
    intensity: float,
    response: int,
    config: StreamingConfig,
) -> StreamingState:
    """Online natural-gradient variational Bayes update (Gaussian approximation).

    Uses a single-step stochastic natural gradient to update the
    mean-field Gaussian approximation q(threshold, slope).
    """
    lr = config.vi_learning_rate
    means = state.vi_means.copy()
    stds = state.vi_stds.copy()

    rng = np.random.default_rng(
        None if config.random_seed is None else config.random_seed + state.step,
    )

    n_samples = 8
    eps = rng.standard_normal((n_samples, 2))
    samples = means + stds * eps  # reparameterization trick
    samples[:, 1] = np.abs(samples[:, 1])

    ll = np.array(
        [_single_log_likelihood(intensity, response, s[0], s[1]) for s in samples]
    )

    # Score-function gradient estimate for means
    grad_means = np.mean(
        ll[:, None] * (eps / stds),
        axis=0,
    )
    # Gradient estimate for log-std
    grad_log_stds = np.mean(
        ll[:, None] * (eps**2 - 1),
        axis=0,
    )

    means = means + lr * grad_means
    log_stds = np.log(stds) + lr * grad_log_stds
    stds = np.exp(np.clip(log_stds, -5.0, 2.0))

    return StreamingState(
        method=state.method,
        step=state.step + 1,
        vi_means=means,
        vi_stds=stds,
        ess_history=state.ess_history,
    )


def _single_log_likelihood(
    intensity: float,
    response: int,
    threshold: float,
    slope: float,
) -> float:
    """Scalar log-likelihood for one particle/sample."""
    p = float(expit(slope * (intensity - threshold)))
    p = max(min(p, 1.0 - 1e-12), 1e-12)
    if response == 1:
        return float(np.log(p))
    return float(np.log1p(-p))


def summarize(state: StreamingState) -> dict[str, float]:
    """Return a summary dict of the current posterior estimate."""
    return {
        "threshold": state.threshold_estimate,
        "slope": state.slope_estimate,
        "ess": state.effective_sample_size,
        "step": float(state.step),
    }
