import numpy as np
from importlib import import_module

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from dask_ml.wrappers import ParallelPostFit

from utils import split_module_class


class Regressor:
    metrics = {
        "MSE": mean_squared_error,
        "R2": r2_score,
        "MAPE": mean_absolute_percentage_error,
    }

    def __init__(
        self,
        model,
        model_parameters={},
        preprocessing=[],
        pre_parameters=[],
    ):
        module_name, class_name = split_module_class(model)
        model_module = import_module(module_name)
        self.model = getattr(model_module, class_name)(**model_parameters)
        pre_stages = []
        for i, stage_module in enumerate(preprocessing):
            module_name, class_name = split_module_class(stage_module)
            model_module = import_module(module_name)
            stage = getattr(model_module, class_name)(**pre_parameters[i])
            pre_stages.append((f"{1}_stage", stage))
        imputer = SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=0.0, copy=False
        )
        if pre_stages:
            self.model = ParallelPostFit(
                Pipeline([("nan_fix", imputer), *pre_stages, ("regression", self.model)])
            )
        else:
            self.model = ParallelPostFit(
                Pipeline([("nan_fix", imputer), ("regression", self.model)])
            )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def compute_metrics(self, y_pred, y_hat):
        results = []
        for metric in self.metrics.values():
            results.append(metric(y_hat, y_pred))
        return results
