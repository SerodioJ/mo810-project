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
        "noise": (0, None),
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
                [("nan_fix", imputer), *pre_stages, ("cluster", self.model)]
            )
        else:
            self.model = Pipeline([("nan_fix", imputer), ("cluster", self.model)])

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    def compute_metrics(self, y_pred, y_hat):
        results = []
        repeated = []
        for t, metric in self.metrics.values():
            if t in repeated:
                continue
            if t == 0:  # Model specific metric
                if hasattr(self.model["cluster"], "inertia_"):
                    results.append(self.model["cluster"].inertia_)
                    results.append(0)
                else:
                    labels = self.model["cluster"].labels_

                    # Number of clusters in labels, ignoring noise if present.
                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                    results.append(n_clusters_)
                    n_noise_ = list(labels).count(-1)
                    results.append(n_noise_)
                repeated.append(t)

            elif t == 1:
                results.append(metric(y_hat, y_pred))
            else:
                repeated.append(t)
                results.extend(metric(y_hat, y_pred))

        return results
