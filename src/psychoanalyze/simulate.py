from dataclasses import dataclass

import numpy as np
import pymc as pm
import xarray as xr


@dataclass
class NormalParams:
    mu: float
    sigma: float


@dataclass
class LogisticPrior:
    x0: NormalParams
    k_sigma: float


def run_prior_predictive(
    n_blocks: int = 1,
    n_trials_per_block: int | None = 100,
    logistic_prior: LogisticPrior = LogisticPrior(
        x0=NormalParams(mu=0, sigma=0.5), k_sigma=2.0
    ),
    draws: int = 500,
    random_seed: int | None = None,
) -> pm.backends.arviz.InferenceData:
    n_trials_per_block = n_trials_per_block or 100
    n_trials = n_blocks * n_trials_per_block

    level_grid = (np.arange(-3, 4) * logistic_prior.x0.sigma) + logistic_prior.x0.mu
    intensities = xr.DataArray(
        np.random.choice(level_grid, size=n_trials),
        dims="trial",
    )
    block_indices = xr.DataArray(
        np.repeat(np.arange(n_blocks), n_trials_per_block),
        dims="trial",
    )

    coords: dict[str, np.ndarray] = {
        "block": np.arange(n_blocks),
        "trial": np.arange(n_trials),
    }

    with pm.Model(coords=coords):
        intensity = pm.Data("x", intensities.values, dims="trial")
        block_id = pm.Data(
            "block_id", block_indices.values.astype(np.int32), dims="trial"
        )

        x0 = pm.Normal(
            "x0",
            mu=logistic_prior.x0.mu,
            sigma=logistic_prior.x0.sigma,
            dims="block",
        )
        k = pm.HalfNormal("k", sigma=logistic_prior.k_sigma, dims="block")
        gamma = pm.Beta("gamma", alpha=1, beta=9, dims="block")
        lam = pm.Beta("lambda", alpha=1, beta=9, dims="block")

        logit_p = pm.invlogit(k[block_id] * (intensity - x0[block_id]))
        p = gamma[block_id] + ((1 - gamma[block_id] - lam[block_id]) * logit_p)
        pm.Bernoulli("y", p=p, dims="trial")

        idata = pm.sample_prior_predictive(draws=draws, random_seed=random_seed)
    return idata
