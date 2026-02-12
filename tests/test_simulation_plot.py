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

def test_psychometric_function_output_range():
    x = 0
    x0 = 0
    k = 1
    p = psychometric_function(x, x0, k)
    assert 0 <= p <= 1

def test_psychometric_function_broadcasting():
    x = np.array([-100, 0, 100])
    x0 = 0
    k = 1
    p = psychometric_function(x, x0, k)
    # Strong assertion covering the entire array behavior
    np.testing.assert_allclose(p, [0, 0.5, 1], atol=1e-5)

def test_extract_simulation_data_merges_all_sources():
    logistic_prior = LogisticPrior(x0=NormalParams(mu=0, sigma=1), k_sigma=1)
    idata = run_prior_predictive(n_blocks=1, n_trials_per_block=10, logistic_prior=logistic_prior)

    merged = _extract_simulation_data(idata, draw=0, chain=0)

    # Assert all expected variable groups are present in the merged dataset
    assert {"x", "x0", "y"}.issubset(merged.data_vars)

def test_extract_simulation_data_reduces_chains():
    logistic_prior = LogisticPrior(x0=NormalParams(mu=0, sigma=1), k_sigma=1)
    idata = run_prior_predictive(n_blocks=1, n_trials_per_block=10, logistic_prior=logistic_prior)

    merged = _extract_simulation_data(idata, draw=0, chain=0)

    # Assert draw and chain dimensions are completely removed
    assert "chain" not in merged.dims and "draw" not in merged.dims

def test_compute_x_range_respects_prior():
    logistic_prior = LogisticPrior(x0=NormalParams(mu=0, sigma=1), k_sigma=1)
    dummy_data = xr.Dataset({"x": (("trial"), [10, 20])})

    x = _compute_x_range(dummy_data, logistic_prior)

    # Assert range is strictly determined by prior (mu +/- 3*sigma)
    expected_min = logistic_prior.x0.mu - 3 * logistic_prior.x0.sigma
    expected_max = logistic_prior.x0.mu + 3 * logistic_prior.x0.sigma
    np.testing.assert_allclose([x.min(), x.max()], [expected_min, expected_max])

def test_compute_x_range_infers_from_data():
    dummy_data = xr.Dataset({"x": (("trial"), [10, 20])})

    x = _compute_x_range(dummy_data, None)

    # Assert range covers the data range with padding
    assert x.min() < 10 and x.max() > 20

def test_compute_theoretical_curve_structure():
    x_vals = np.linspace(-3, 3, 10)
    merged_data = xr.Dataset(
        {
            "x0": (("block"), [0]),
            "k": (("block"), [1]),
            "gamma": (("block"), [0]),
            "lambda": (("block"), [0])
        },
        coords={"block": [0]}
    )

    curve_df = _compute_theoretical_curve(x_vals, merged_data)

    # Assert strictly required columns are present
    assert list(curve_df.columns) == ["block", "x", "p"]

def test_compute_theoretical_curve_values():
    x_vals = np.linspace(-3, 3, 10)
    merged_data = xr.Dataset(
        {
            "x0": (("block"), [0]),
            "k": (("block"), [100]), # Steep slope
            "gamma": (("block"), [0]),
            "lambda": (("block"), [0])
        },
        coords={"block": [0]}
    )

    curve_df = _compute_theoretical_curve(x_vals, merged_data)

    # Assert behavior: steep slope should yield high probability for x > x0
    idx_positive = (curve_df['x'] > 0.1)
    assert (curve_df.loc[idx_positive, "p"] > 0.99).all()

def test_aggregate_simulation_data_groups_correctly():
    merged_data = xr.Dataset(
        {
            "x": (("trial"), [1, 2]),
            "block_id": (("trial"), [0, 1]),
            "y": (("trial"), [0, 1])
        }
    )

    summary_df = _aggregate_simulation_data(merged_data)

    # Assert unique groups are preserved
    expected_groups = {(0, 1), (1, 2)} # (block, x) tuples
    actual_groups = set(zip(summary_df["block"], summary_df["x"]))
    assert actual_groups == expected_groups

def test_aggregate_simulation_data_calculates_mean():
    merged_data = xr.Dataset(
        {
            "x": (("trial"), [1, 1]),
            "block_id": (("trial"), [0, 0]),
            "y": (("trial"), [0, 1])
        }
    )

    summary_df = _aggregate_simulation_data(merged_data)

    # Assert mean calculation is correct (0+1)/2 = 0.5
    np.testing.assert_allclose(summary_df["p"].values, [0.5])

def test_aggregate_simulation_data_calculates_count():
    merged_data = xr.Dataset(
        {
            "x": (("trial"), [1, 1, 1]),
            "block_id": (("trial"), [0, 0, 0]),
            "y": (("trial"), [0, 1, 0])
        }
    )

    summary_df = _aggregate_simulation_data(merged_data)

    # Assert count is correct
    assert summary_df["n"].values[0] == 3

def test_plot_prior_simulation_returns_chart():
    logistic_prior = LogisticPrior(
        x0=NormalParams(mu=0, sigma=1),
        k_sigma=1
    )
    idata = run_prior_predictive(
        n_blocks=1,
        n_trials_per_block=5,
        logistic_prior=logistic_prior
    )

    chart = plot_prior_simulation(idata, logistic_prior=logistic_prior)

    assert isinstance(chart, alt.LayerChart)
