"""PsychoAnalyze dashboard as a marimo notebook.

Interactive data simulation & analysis for psychophysics.
Replaces the former Dash dashboard (removed).
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full", app_title="PsychoAnalyze")

with app.setup:
    import psychoanalyze as psy


@app.cell
def _():
    import xarray as xr
    import altair as alt
    import numpy as np
    from scipy.special import expit
    import marimo as mo

    import arviz_plots as azp
    from psychoanalyze.plotting.psychometric import plot_prior_simulation

    return alt, azp, expit, mo, np, plot_prior_simulation, xr


@app.cell
def _(mo):
    x0_mu = mo.ui.number(label="x0_mu", value=0)
    x0_sigma = mo.ui.number(label="x0_sigma", value=1)
    k_sigma = mo.ui.number(label="k_sigma", value=1)
    n_blocks = mo.ui.number(label="n_blocks", value=1)
    n_trials_per_block = mo.ui.number(label="n_trials_per_block", value=50)
    mo.vstack([x0_mu, x0_sigma, k_sigma, n_blocks, n_trials_per_block])
    return k_sigma, n_blocks, n_trials_per_block, x0_mu, x0_sigma


@app.cell
def _(k_sigma, x0_mu, x0_sigma):
    logistic_prior = psy.simulate.LogisticPrior(
        x0=psy.simulate.NormalParams(
            mu=x0_mu.value,
            sigma=x0_sigma.value,
        ),
        k_sigma=k_sigma.value,
    )
    return (logistic_prior,)


@app.cell
def _(logistic_prior, n_blocks, n_trials_per_block):
    prior_samples = psy.simulate.run_prior_predictive(
        n_blocks=int(n_blocks.value),
        n_trials_per_block=int(n_trials_per_block.value),
        logistic_prior=logistic_prior,
    )
    return (prior_samples,)


@app.cell
def _(logistic_prior, plot_prior_simulation, prior_samples):
    plot_prior_simulation(prior_samples, logistic_prior=logistic_prior)
    return


@app.cell
def _(azp, prior_samples):
    pc = azp.plot_dist(
        prior_samples,
        var_names=["x0", "k", "gamma", "lambda"],
        kind="ecdf",
        group="prior",
    )
    pc.show()
    return


if __name__ == "__main__":
    app.run()
