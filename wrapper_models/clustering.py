import numpy as np
from importlib import import_module

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    rand_score,
)

from utils import split_module_class


class Cluster:

    metrics = {
        "homogeneity": (2, homogeneity_completeness_v_measure),
        "completeness": (2, homogeneity_completeness_v_measure),
        "v_measure": (2, homogeneity_completeness_v_measure),
        "rand": (1, rand_score),
        "inertia": (0, None),
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
            self.model = Pipeline(
                [("nan_fix", imputer), *pre_stages, ("regresion", self.model)]
            )
        else:
            self.model = Pipeline([("nan_fix", imputer), ("regresion", self.model)])

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    def compute_metrics(self, y_pred, y_hat):
        results = []
        repeated = []
        for t, metric in self.metrics.values():
            if t == 0:
                if hasattr(self.model, "inertia__"):
                    results.append(self.model.inertia_)
                else:
                    results.append(0)
            elif t == 1:
                results.append(metric(y_hat, y_pred))
            else:
                if t in repeated:
                    continue
                repeated.append(t)
                results.extend(metric(y_hat, y_pred))

        return results
