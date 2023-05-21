import numpy as np
from importlib import import_module

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, homogeneity_score
from dask_ml.wrappers import ParallelPostFit

from utils import split_module_class


class Cluster:

    metrics = {
        "homogeneity": homogeneity_score,
        "f1": f1_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
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
                Pipeline([("nan_fix", imputer), *pre_stages, ("regresion", self.model)])
            )
        else:
            self.model = ParallelPostFit(
                Pipeline([("nan_fix", imputer), ("regresion", self.model)])
            )

    def fit(self, X, y):
        self.model.fit(X)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    @classmethod
    def compute_metrics(cls, y_pred, y_hat):
        results = []
        for metric in cls.metrics.values():
            results.append(metric(y_hat, y_pred))
        return results
