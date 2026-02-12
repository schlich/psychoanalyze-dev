import pytest
import altair as alt
import numpy as np
import xarray as xr
from psychoanalyze.simulate import run_prior_predictive, LogisticPrior, NormalParams, psychometric_function
from psychoanalyze.plotting.psychometric import (
    plot_prior_simulation,
    _extract_simulation_data,
    _compute_x_range,
    _compute_theoretical_curve,
    _aggregate_simulation_data
)

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

def test_extract_simulation_data():
    logistic_prior = LogisticPrior(x0=NormalParams(mu=0, sigma=1), k_sigma=1)
    idata = run_prior_predictive(n_blocks=1, n_trials_per_block=10, logistic_prior=logistic_prior)

    # Test merging of prior, constant_data, and prior_predictive
    merged = _extract_simulation_data(idata, draw=0, chain=0)
    assert "x" in merged # from constant_data
    assert "x0" in merged # from prior
    assert "y" in merged # from prior_predictive

    # Verify scalar reduction for parameters
    assert merged["x0"].ndim == 1 # only block dim remains, draw/chain are gone

def test_compute_x_range():
    logistic_prior = LogisticPrior(x0=NormalParams(mu=0, sigma=1), k_sigma=1)

    # Test with logistic_prior provided
    dummy_data = xr.Dataset({"x": (("trial"), [10, 20])})
    x = _compute_x_range(dummy_data, logistic_prior)
    assert np.isclose(x.min(), -3)
    assert np.isclose(x.max(), 3)

    # Test inferred from data
    x = _compute_x_range(dummy_data, None)
    assert x.min() < 10
    assert x.max() > 20

def test_compute_theoretical_curve():
    # Setup dummy merged data with multiple blocks
    x_vals = np.linspace(-3, 3, 10)
    merged_data = xr.Dataset(
        {
            "x0": (("block"), [0, 0]),
            "k": (("block"), [1, 100]), # One gentle slope, one steep
            "gamma": (("block"), [0, 0]),
            "lambda": (("block"), [0, 0])
        },
        coords={"block": [0, 1]}
    )

    curve_df = _compute_theoretical_curve(x_vals, merged_data)

    # Check structure
    assert "p" in curve_df.columns
    assert "x" in curve_df.columns
    assert "block" in curve_df.columns

    # Check that we have data for both blocks
    assert len(curve_df["block"].unique()) == 2

    # Check values for steep slope (k=100) at x=0.1 (should be ~1)
    steep_block = curve_df[curve_df["block"] == 1]
    # find closest x to 0.1
    idx = (steep_block['x'] - 0.1).abs().idxmin()
    assert steep_block.loc[idx, "p"] > 0.9

def test_aggregate_simulation_data():
    # Setup dummy data
    merged_data = xr.Dataset(
        {
            "x": (("trial"), [1, 1, 2, 2]),
            "block_id": (("trial"), [0, 0, 1, 1]),
            "y": (("trial"), [0, 1, 1, 1])
        }
    )

    summary_df = _aggregate_simulation_data(merged_data)

    # Check grouping
    # Block 0, x=1: mean=0.5, count=2
    b0 = summary_df[(summary_df["block"] == 0) & (summary_df["x"] == 1)].iloc[0]
    assert b0["p"] == 0.5
    assert b0["n"] == 2

    # Block 1, x=2: mean=1.0, count=2
    b1 = summary_df[(summary_df["block"] == 1) & (summary_df["x"] == 2)].iloc[0]
    assert b1["p"] == 1.0
    assert b1["n"] == 2

def test_plot_prior_simulation_integration():
    logistic_prior = LogisticPrior(
        x0=NormalParams(mu=0, sigma=1),
        k_sigma=1
    )
    idata = run_prior_predictive(
        n_blocks=2,
        n_trials_per_block=10,
        logistic_prior=logistic_prior
    )

    chart = plot_prior_simulation(idata, logistic_prior=logistic_prior)
    assert isinstance(chart, alt.LayerChart)
