import pymc as pm
import numpy as np
import pytest
import arviz as az
from psychoanalyze.testing.pymc import assert_converged, assert_no_divergences, verify_model_structure

def test_verify_model_structure():
    with pm.Model() as model:
        pm.Normal("x", mu=0, sigma=1)

    verify_model_structure(model)

    with pm.Model() as empty_model:
        pass

    with pytest.raises(AssertionError, match="Model has no free random variables"):
        verify_model_structure(empty_model)

def test_assert_converged():
    # Create a fake InferenceData with good convergence
    # 4 chains, 1000 draws
    idata = az.from_dict(
        posterior={
            "x": np.random.randn(4, 1000),
        }
    )
    assert_converged(idata)

    # Create a fake InferenceData with bad convergence (split chains with different means)
    bad_posterior = np.vstack([
        np.random.randn(2, 1000),
        np.random.randn(2, 1000) + 100
    ])
    idata_bad = az.from_dict(posterior={"x": bad_posterior})

    with pytest.raises(AssertionError, match="R-hat check failed"):
        assert_converged(idata_bad, rhat_tol=1.01)

def test_assert_no_divergences():
    # Good case
    sample_stats = {
        "diverging": np.zeros((4, 100), dtype=bool)
    }
    idata = az.from_dict(sample_stats=sample_stats)
    assert_no_divergences(idata)

    # Bad case
    sample_stats_bad = {
        "diverging": np.zeros((4, 100), dtype=bool)
    }
    sample_stats_bad["diverging"][0, 10] = True
    idata_bad = az.from_dict(sample_stats=sample_stats_bad)

    with pytest.raises(AssertionError, match="Found 1 divergent transitions"):
        assert_no_divergences(idata_bad)
