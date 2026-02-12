import altair as alt
import numpy as np
import pandas as pd
import xarray as xr
from psychoanalyze.simulate import LogisticPrior, psychometric_function


def plot_prior_simulation(
    idata, logistic_prior: LogisticPrior | None = None, draw: int = 0, chain: int = 0
) -> alt.LayerChart:
    """Plots a single simulation from the prior predictive distribution.

    Args:
        idata: The InferenceData object containing prior samples and constant data.
        logistic_prior: The prior parameters used for simulation, used to determine
            the x-axis range for the psychometric curve. If None, the range is
            inferred from the data.
        draw: The draw index to plot.
        chain: The chain index to plot.

    Returns:
        An Altair LayerChart containing the theoretical psychometric curve and
        the simulated data points (aggregated).
    """
    # Extract data for the specified draw and chain
    # We merge prior, constant_data, and prior_predictive (if present) to ensure
    # we have parameters, inputs, and simulated outcomes.
    ds_list = [idata.prior.sel(draw=draw, chain=chain), idata.constant_data]
    if hasattr(idata, "prior_predictive"):
        ds_list.append(idata.prior_predictive.sel(draw=draw, chain=chain))

    merged = xr.merge(ds_list)

    # Determine x range for the curve
    if logistic_prior is not None:
        x_start = logistic_prior.x0.mu - 3 * logistic_prior.x0.sigma
        x_end = logistic_prior.x0.mu + 3 * logistic_prior.x0.sigma
        x = np.linspace(x_start, x_end, 100)
    else:
        # Infer from data if not provided
        x_vals = merged["x"].values
        x_min = float(x_vals.min())
        x_max = float(x_vals.max())
        padding = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
        x = np.linspace(x_min - padding, x_max + padding, 100)

    # Calculate theoretical curve
    # Use xarray broadcasting to handle multiple blocks
    x_da = xr.DataArray(x, coords={"x": x}, dims="x")

    # merged["lambda"] is a keyword conflict if accessed as attribute
    lam = merged["lambda"]

    # Calculate p using the psychometric function
    # parameters in merged have dim 'block'
    y_da = psychometric_function(x_da, merged.x0, merged.k, merged.gamma, lam)

    # Convert to DataFrame for Altair
    # y_da has dims (block, x)
    curve_df = y_da.to_dataframe(name="p").reset_index()

    # Plotting curve
    fit_line = (
        alt.Chart(curve_df)
        .mark_line()
        .encode(x="x", y="p", color="block:N")
    )

    # Plotting data points
    # Convert to pandas DataFrame for easy grouping
    # We select only relevant variables to avoid broadcasting issues with others
    points_df = merged[["x", "block_id", "y"]].to_dataframe().reset_index(drop=True)

    # Aggregate
    summary_df = (
        points_df.groupby(["x", "block_id"])
        .agg(p=("y", "mean"), n=("y", "count"))
        .reset_index()
        .rename(columns={"block_id": "block"})
    )

    data_points = (
        alt.Chart(summary_df)
        .mark_point()
        .encode(x="x", y="p", size="n", color=alt.Color("block:N"))
    )

    return fit_line + data_points
