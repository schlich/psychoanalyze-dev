import numpy as np
import xarray as xr

from psychoanalyze import plot, psi, simulate


class TestPsychometricFunction:
    def test_psi_plot_empty_dataframe(self):
        points = xr.Dataset(
            {
                "hit_rate": ("sample", []),
                "magnitude": ("sample", []),
            }
        )

        fig = psi.plot(points)
        assert fig.layout.yaxis.title.text == "hit_rate"
        assert fig.layout.xaxis.title.text == "magnitude"


class TestPlotsEntrypoint:
    def test_plot_all_returns_four_target_plots(self) -> None:
        trials = xr.Dataset(
            {
                "Magnitude": ("trial", [1.0, 2.0]),
                "Result": ("trial", [True, False]),
            }
        )

        plots = plot.all(trials)

        assert set(plots.keys()) == {
            "Psychometric function",
            "Threshold vs Time",
            "Strength-duration",
            "Weber curves",
        }


class TestDataGenerationSimulation:
    """BDD: Data generation and simulation (prior predictive) feature."""

    def test_run_prior_predictive_sampling_with_default_draws(self) -> None:
        """Run prior predictive sampling with default draws."""
        idata = simulate.run_prior_predictive(draws=20, random_seed=42)
        assert "prior" in idata.groups()
        prior = idata["prior"]
        assert prior.sizes.get("draw", 0) > 0
        assert len(prior.data_vars) > 0
        # Prior predictive group may be present or folded into prior depending on PyMC version
        assert (
            "prior_predictive" in idata.groups()
            or "obs" in prior.data_vars
            or "y" in prior.data_vars
        )

    def test_run_prior_predictive_with_specified_draws(self) -> None:
        """Run prior predictive sampling with a specified number of draws."""
        idata = simulate.run_prior_predictive(draws=50, random_seed=123)
        assert "prior" in idata.groups()
        assert idata["prior"].sizes["draw"] == 50

    def test_prior_predictive_curves_bounded(self) -> None:
        """Prior predictive hit rates and params are bounded and finite."""
        idata = simulate.run_prior_predictive(draws=30, random_seed=456)
        x0 = idata["prior"]["x0"].values
        k = idata["prior"]["k"].values
        assert np.all(np.isfinite(x0))
        assert np.all(k > 0)
        if "prior_predictive" in idata.groups() and "obs" in idata["prior_predictive"].data_vars:
            obs = idata["prior_predictive"]["obs"].values
            assert np.all((obs >= 0) & (obs <= 1))
