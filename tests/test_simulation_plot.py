import pytest
import altair as alt
import numpy as np
import xarray as xr
from psychoanalyze.simulate import run_prior_predictive, LogisticPrior, NormalParams, psychometric_function
from psychoanalyze.plotting.psychometric import plot_prior_simulation

def test_psychometric_function():
    # Test basic output range
    x = 0
    x0 = 0
    k = 1
    p = psychometric_function(x, x0, k)
    assert p == 0.5

    # Test broadcasting
    x = np.array([-100, 0, 100])
    p = psychometric_function(x, x0, k)
    assert np.allclose(p, [0, 0.5, 1])

def test_plot_prior_simulation():
    logistic_prior = LogisticPrior(
        x0=NormalParams(mu=0, sigma=1),
        k_sigma=1
    )
    # n_trials_per_block must be enough to not fail sampling maybe?
    # But prior predictive is simple generation.
    idata = run_prior_predictive(
        n_blocks=2,
        n_trials_per_block=10,
        logistic_prior=logistic_prior
    )

    chart = plot_prior_simulation(idata, logistic_prior=logistic_prior)
    assert isinstance(chart, alt.LayerChart)

    # Test without prior provided (infer range)
    chart = plot_prior_simulation(idata)
    assert isinstance(chart, alt.LayerChart)
