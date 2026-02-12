import numpy as np
import pandas as pd
import pymc as pm
import pytest
from psychoanalyze.model_builder import ModelBuilder
import xarray as xr
import os

class LinearModel(ModelBuilder):
    _model_type = "LinearModel"
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self._generate_and_preprocess_model_data(X, y)

        with pm.Model(coords=self.model_coords) as self.model:
            # Check if X is DataFrame or numpy array, handle accordingly
            if isinstance(self.X, pd.DataFrame):
                x_val = self.X["x"].values
            else:
                x_val = self.X[:, 0]

            x_data = pm.Data("x_data", x_val, dims="obs")

            # Similar check for y
            if isinstance(self.y, pd.Series):
                y_val = self.y.values
            else:
                y_val = self.y

            y_data = pm.Data("y_data", y_val, dims="obs")

            slope = pm.Normal("slope", mu=0, sigma=1)
            intercept = pm.Normal("intercept", mu=0, sigma=1)
            sigma = pm.HalfNormal("sigma", sigma=1)

            mu = slope * x_data + intercept
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_data, dims="obs")

    def _generate_and_preprocess_model_data(self, X, y):
        self.X = X
        if isinstance(y, (pd.Series, np.ndarray)):
             self.y = pd.Series(y, name="y")
        else:
             self.y = y
        self.model_coords = {"obs": np.arange(len(X))}

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self):
        return self.model_config

    def _data_setter(self, X, y=None):
        with self.model:
            if isinstance(X, pd.DataFrame):
                x_val = X["x"].values
            else:
                # If it's a numpy array or something else, assume it matches expected shape
                # This simplistic handling is just for the test model
                x_val = X[:, 0] if hasattr(X, "shape") and len(X.shape) > 1 else X

            pm.set_data({"x_data": x_val})
            if y is not None:
                if isinstance(y, pd.Series):
                    y_val = y.values
                else:
                    y_val = y
                pm.set_data({"y_data": y_val})

    @staticmethod
    def get_default_model_config():
        return {}

    @staticmethod
    def get_default_sampler_config():
        return {"draws": 20, "tune": 10, "chains": 1, "target_accept": 0.8}

def test_model_builder_lifecycle(tmp_path):
    X = pd.DataFrame({"x": np.linspace(0, 10, 20)})
    y = 2 * X["x"] + 1 + np.random.normal(0, 0.5, 20)

    model = LinearModel()
    idata = model.fit(X, y)

    assert "posterior" in idata
    assert "prior" in idata

    # Test predict
    pred = model.predict(X)
    assert pred.shape == (20,)

    # Test save and load
    save_path = tmp_path / "model.nc"
    model.save(str(save_path))

    loaded_model = LinearModel.load(str(save_path))
    assert loaded_model.id == model.id

    # Test prediction with loaded model
    pred_loaded = loaded_model.predict(X)
    assert pred_loaded.shape == (20,)
