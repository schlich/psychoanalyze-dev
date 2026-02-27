"""Type definitions for PsychoAnalyze Bayesian analysis pipeline."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class LinkFunction(enum.Enum):
    """Psychometric link functions for sigmoid fitting."""

    LOGIT = "logit"
    PROBIT = "probit"
    WEIBULL = "weibull"
    GUMBEL = "gumbel"


class InferenceMethod(enum.Enum):
    """Supported Bayesian inference backends."""

    MCMC = "mcmc"
    SMC = "smc"
    ADVI = "advi"
    FULLRANK_ADVI = "fullrank_advi"
    PATHFINDER = "pathfinder"


@dataclass
class BayesianFitSettings:
    """Configuration for a Bayesian psychometric fit."""

    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95
    inference_method: InferenceMethod = InferenceMethod.MCMC
    random_seed: int | None = None


@dataclass
class StreamingConfig:
    """Configuration for streaming Bayesian updates.

    Supports sequential Monte Carlo (SMC) and variational inference (VI)
    backends for real-time processing of incoming trial data.
    """

    method: InferenceMethod = InferenceMethod.SMC
    n_particles: int = 1000
    ess_threshold: float = 0.5
    vi_max_iter: int = 10000
    vi_learning_rate: float = 0.01
    random_seed: int | None = None


@dataclass
class FitArtifacts:
    """Artifacts produced by a Bayesian fit."""

    threshold: float
    slope: float
    threshold_hdi: tuple[float, float] = (float("nan"), float("nan"))
    slope_hdi: tuple[float, float] = (float("nan"), float("nan"))
    converged: bool = False
    n_divergences: int = 0


@dataclass
class StreamingState:
    """Mutable state for a streaming Bayesian estimator.

    Tracks particle populations (SMC) or variational parameters (VI)
    across sequential observations.
    """

    method: InferenceMethod = InferenceMethod.SMC
    step: int = 0
    log_weights: NDArray[np.floating] = field(
        default_factory=lambda: np.array([], dtype=np.float64),
    )
    particles: NDArray[np.floating] = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64),
    )
    vi_means: NDArray[np.floating] = field(
        default_factory=lambda: np.array([0.0, 1.0]),
    )
    vi_stds: NDArray[np.floating] = field(
        default_factory=lambda: np.array([1.0, 1.0]),
    )
    ess_history: list[float] = field(default_factory=list)

    @property
    def threshold_estimate(self) -> float:
        """Current threshold point estimate (weighted mean of particles or VI mean)."""
        if self.method == InferenceMethod.SMC and len(self.particles) > 0:
            weights = _softmax(self.log_weights)
            return float(np.dot(weights, self.particles[:, 0]))
        return float(self.vi_means[0])

    @property
    def slope_estimate(self) -> float:
        """Current slope point estimate."""
        if self.method == InferenceMethod.SMC and len(self.particles) > 0:
            weights = _softmax(self.log_weights)
            return float(np.dot(weights, self.particles[:, 1]))
        return float(self.vi_means[1])

    @property
    def effective_sample_size(self) -> float:
        """ESS of the current particle population."""
        if len(self.log_weights) == 0:
            return 0.0
        weights = _softmax(self.log_weights)
        return float(1.0 / np.sum(weights**2))


def _softmax(log_w: NDArray[np.floating]) -> NDArray[np.floating]:
    """Numerically stable softmax for log-weights."""
    shifted = log_w - np.max(log_w)
    exp_w = np.exp(shifted)
    return exp_w / np.sum(exp_w)


class CacheBackend(Protocol):
    """Protocol for pluggable fit-artifact caching."""

    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes) -> None: ...
