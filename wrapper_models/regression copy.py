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
                Pipeline([("nan_fix", imputer), *pre_stages, ("regresion", self.model)])
            )
        else:
            self.model = ParallelPostFit(
                Pipeline([("nan_fix", imputer), ("regresion", self.model)])
            )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    @classmethod
    def compute_metrics(cls, y_pred, y_hat):
        results = []
        for metric in cls.metrics.values():
            results.append(metric(y_pred, y_hat))
        return results

    # def __init__(


#         self,
#         model,
#         model_parameters={},
#         window_size=1,
#         preprocessing=[],
#         pre_parameters=[],
#     ):
#         module_name, class_name = split_module_class(model)
#         model_module = import_module(module_name)
#         self.model = getattr(model_module, class_name)(**model_parameters)
#         pre_stages = []
#         for i, stage_module in enumerate(preprocessing):
#             module_name, class_name = split_module_class(stage_module)
#             model_module = import_module(module_name)
#             stage = getattr(model_module, class_name)(**pre_parameters[i])
#             pre_stages.append((f"{1}_stage", stage))
#         if pre_stages:
#             self.model = Pipeline([*pre_stages, ("regresion", self.model)])
#         self.window_size = window_size

#     def fit(self, X, y):
#         X, y = self.fit_reshape(X, y, self.window_size)
#         self.model.fit(X, y)

#     def predict(self, X):
#         shape = X.shape
#         X = self.predict_reshape(X, self.window_size)
#         y_predict = self.model.predict(X)
#         return y_predict.reshape(shape)

#     def metrics(self, y_pred, y_hat):
#         metrics_dict = {}
#         metrics_dict["mse"] = mean_squared_error(y_hat, y_pred)
#         metrics_dict["r2_score"] = r2_score(y_hat, y_pred)
#         return metrics_dict

#     @staticmethod
#     def fit_reshape(X, y, window_size):
#         X = np.lib.stride_tricks.sliding_window_view(X, window_size, -1)
#         X = X.reshape(-1, X.shape[-1])
#         padding = window_size // 2
#         y = y[:, padding : y.shape[1] - padding]
#         y = np.lib.stride_tricks.sliding_window_view(y, 1, -1)
#         y = y.flatten()
#         print(y.shape)
#         return X, y

#     @staticmethod
#     def predict_reshape(X, window_size):
#         padding = window_size // 2
#         npad = [(0, 0)] * len(X.shape)
#         npad[-1] = (padding, padding)
#         X = np.pad(X, pad_width=npad, mode="constant", constant_values=0)
#         X = np.lib.stride_tricks.sliding_window_view(X, window_size, -1)
#         X = X.reshape(-1, X.shape[-1])
#         return X
