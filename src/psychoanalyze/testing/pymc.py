"""Testing utilities for PyMC models."""

import arviz as az
import pymc as pm


def assert_converged(idata: az.InferenceData, rhat_tol: float = 1.05, ess_tol: float = 100) -> None:
    """
    Assert that the model has converged.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object containing the posterior samples.
    rhat_tol : float, optional
        Tolerance for R-hat statistic. Defaults to 1.05.
    ess_tol : float, optional
        Minimum acceptable Effective Sample Size. Defaults to 100.

    Raises
    ------
    AssertionError
        If R-hat is greater than rhat_tol or ESS is less than ess_tol.
    """
    if "posterior" not in idata:
        raise ValueError("InferenceData does not contain posterior samples.")

    rhat = az.rhat(idata)
    ess = az.ess(idata)

    # Check R-hat
    rhat_max = rhat.max().to_array().max().item()
    if rhat_max > rhat_tol:
        raise AssertionError(f"R-hat check failed. Max R-hat: {rhat_max} > {rhat_tol}")

    # Check ESS
    ess_min = ess.min().to_array().min().item()
    if ess_min < ess_tol:
        raise AssertionError(f"ESS check failed. Min ESS: {ess_min} < {ess_tol}")


def assert_no_divergences(idata: az.InferenceData) -> None:
    """
    Assert that there are no divergent transitions in the posterior samples.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object containing the sample stats.

    Raises
    ------
    AssertionError
        If there are any divergent transitions.
    """
    if "sample_stats" not in idata:
        # Some samplers might not have sample_stats
        return

    if hasattr(idata.sample_stats, "diverging"):
        divergences = idata.sample_stats.diverging.sum().item()
        if divergences > 0:
            raise AssertionError(f"Found {divergences} divergent transitions.")


def verify_model_structure(model: pm.Model) -> None:
    """
    Perform basic checks on the model structure.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to check.

    Raises
    ------
    AssertionError
        If basic checks fail.
    """
    if not model.free_RVs:
        raise AssertionError("Model has no free random variables.")

    # Check if we can compute logp
    try:
        model.compile_logp()
    except Exception as e:
        raise AssertionError(f"Failed to compile logp: {e}")
